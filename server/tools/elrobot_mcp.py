"""MCP server: lets Claude (Claude Code / Claude Desktop) or Codex see through the robot's cameras and drive the
ElRobot follower arm through NormaCore Station. Plain JSON-RPC over stdio, no extra dependencies.

    Claude Code: registered in .mcp.json at the project root (server name "elrobot")
    Codex:       ~/.codex/config.toml ->  [mcp_servers.elrobot]  command = "/abs/path/tools/elrobot_mcp"

All motion goes through tools/arm_control.py (speed limit, desk check, collision stop, max distance per call).
"""
import asyncio
import base64
import json
import logging
import sys
import urllib.request
import subprocess
import socket
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arm_control  # noqa: E402
import vision  # noqa: E402


def _log(text, data):
    """Show Claude's tool calls in the dashboard's reasoning stream. Never let logging break a tool call."""
    try:
        vision.trace("claude", text, data=data)
    except Exception:
        pass

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

INSTRUCTIONS = """You control a real 7-DoF robot arm (NormaCore ElRobot follower, 2-finger gripper) through NormaCore Station.
Work in a closed loop: look -> one small move -> look again. Never chain several moves without looking.
Coordinates are centimetres in the robot base frame: origin = centre of the arm's base on the table, +z up,
+y = straight out in front of the base, x = sideways (check the sign with a small move + look before relying on it).
The 'fingertips' point is the centre of the two fingertips (the spot that touches the object). Home = folded upright,
gripper camera looking across the table: a good first overview. To grasp: call ready, find the object in the table
camera, move above it (orient='down'), descend while checking the gripper camera (object centred between the jaws),
open before the final descent, close with set_gripper(0), then lift. The jaws close along the gripper camera's
image left-right axis: if an object is long, rotate wrist_roll until its long side looks vertical in the gripper image.
Moves land 1-3 cm off (gravity sag, worse when stretched far out): always correct with what the cameras show.
Directions learned on the real arm: +x = the arm's right; +y = away from the base.
Fast pick-and-place (~30 s): pick(color) -> place_at(x, y) -> go_home -> look to verify. If pick can't find the object
in the table camera, look, move_to(x, y, 10) roughly above it yourself, then grasp(color).
The white paper is around x = -17, y = 27 (on the arm's left). For a task that matches a learned skill (run_skill),
you may try the skill, but always verify with the images. Moves are refused if they are too long
(15 cm max) or would hit the table; a blocked joint stops the move. A human with a hand on the power plug supervises.
To show the human what is happening (3D twin, cameras, detections, reasoning), call open_dashboard and give them the link.
"""

# Learned skills (ACT policies trained from teleop demos): first existing checkpoint wins (newest first)
SKILLS = {
    "ball_to_paper": {"seconds": 40, "checkpoints": [
        "training/runs/ball28_act/checkpoints/last/pretrained_model",
        "training/runs/ball18_act/checkpoints/last/pretrained_model",
        "training/runs/ball18_act/checkpoints/004000/pretrained_model"]},
}


def run_skill(name):
    import subprocess
    root = arm_control.ROOT
    ckpt = next((root / c for c in SKILLS[name]["checkpoints"] if (root / c).exists()), None)
    if ckpt is None:
        return f"REFUSED: skill {name} is not trained yet"
    r = subprocess.run([str(root / "tools/run_act"), str(ckpt), "--auto", "--seconds", str(SKILLS[name]["seconds"])],
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=180)
    tail = [l for l in (r.stdout + r.stderr).splitlines() if l.strip()][-4:]
    return f"skill {name} ran ({ckpt.parts[-4]}), exit {r.returncode}. Last output:\n" + "\n".join(tail)


ORIENT = {"type": "string", "enum": ["keep", "down", "free"], "default": "keep",
          "description": "keep = keep the gripper's current direction, down = point straight down (best for grasping "
                         "from above), free = any orientation (most reachable)"}
TOOLS = [
    {"name": "look", "description": "Current images from the table camera and the gripper camera, plus the arm state "
     "(fingertip position, gripper opening, joint angles, motor health).", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "get_state", "description": "Arm state as text only (no images).", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "move_by", "description": "Move the fingertips by dx/dy/dz centimetres (robot base frame) along a straight "
     "line. Max 15 cm per call. Returns the new state and both camera images.",
     "inputSchema": {"type": "object", "properties": {"dx": {"type": "number"}, "dy": {"type": "number"},
                     "dz": {"type": "number"}, "orient": ORIENT}, "required": ["dx", "dy", "dz"]}},
    {"name": "move_in_view", "description": "Move relative to what the GRIPPER CAMERA shows (easiest for centring an "
     "object between the jaws): right_cm toward the image's right edge, up_cm toward the image's top edge, down_cm "
     "straight down (negative = up). These directions rotate with wrist_roll; this tool accounts for it.",
     "inputSchema": {"type": "object", "properties": {"right_cm": {"type": "number"}, "up_cm": {"type": "number"},
                     "down_cm": {"type": "number"}, "orient": ORIENT}, "required": ["right_cm", "up_cm", "down_cm"]}},
    {"name": "move_to", "description": "Move the fingertips to x/y/z centimetres (z = height above the table). Max 15 cm "
     "from the current position. Returns the new state and both camera images.",
     "inputSchema": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"},
                     "z": {"type": "number"}, "orient": {**ORIENT, "default": "down"}}, "required": ["x", "y", "z"]}},
    {"name": "rotate_joint", "description": "Rotate one joint by a number of degrees (max 45). Joints: base (turns the "
     "whole arm left/right), shoulder, elbow, wrist_flex, forearm_roll, wrist_pitch, wrist_roll (rotates the gripper "
     "about its own axis, use it to align the fingers with an object).",
     "inputSchema": {"type": "object", "properties": {"joint": {"type": "string", "enum": arm_control.JOINT_NAMES[:7]},
                     "degrees": {"type": "number"}}, "required": ["joint", "degrees"]}},
    {"name": "set_gripper", "description": "Open/close the gripper: 0 = closed, 100 = fully open. When closing on an "
     "object it stops and squeezes gently, and reports that it is holding something.",
     "inputSchema": {"type": "object", "properties": {"open_percent": {"type": "number"}}, "required": ["open_percent"]}},
    {"name": "ready", "description": "Go to the ready pose: fingertips 12 cm above the table, 22 cm in front of the "
     "base, pointing down. Use it at the start and whenever lost.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "pick", "description": "FASTEST way to pick something up (one call, ~20 s): finds the object by colour in "
     "the TABLE camera, moves above it, then does the automatic camera-guided grasp and lifts it. Then call place_at.",
     "inputSchema": {"type": "object", "properties": {
         "color": {"type": "string", "enum": ["orange", "red", "yellow", "white"], "default": "orange"},
         "grasp_height_cm": {"type": "number", "default": 2.5,
                             "description": "fingertip height for the grip, about half the object's height"}}}},
    {"name": "put_ball_on_cube", "description": "Put the orange ball ON TOP of the Rubik's cube: table camera finds the "
     "cube, the ball is picked, carried high, the GRIPPER camera confirms/corrects the cube's position from straight "
     "above, then the ball is set down 0.5 cm above the top (not dropped). Checks the result with the table camera.",
     "inputSchema": {"type": "object", "properties": {"cube_height_cm": {"type": "number", "default": 5.7}}}},
    {"name": "put_ball_in_mug", "description": "Whole task in one call: hover over the chosen mug with the empty gripper "
     "and lock its position with the gripper camera, pick the orange ball, carry it over the mug 4 cm above the rim and "
     "drop it (never touches the mug), go home, check with the table camera. Every step is shown live on the vision "
     "dashboard (port 8893).",
     "inputSchema": {"type": "object", "properties": {
         "mug": {"type": "string", "enum": ["white mug", "black mug"], "default": "white mug"},
         "mug_height_cm": {"type": "number", "default": 10.0}}}},
    {"name": "grasp", "description": "AUTOMATIC camera-guided grasp (fast, no step-by-step needed): first move the "
     "gripper roughly above the object (within ~5 cm, object visible in the gripper camera, e.g. move_to x,y,12). It "
     "centres the object between the jaws using the gripper camera, turns the wrist for long objects, descends, "
     "closes, lifts 6 cm and reports GRASPED or MISSED.",
     "inputSchema": {"type": "object", "properties": {
         "color": {"type": "string", "enum": ["orange", "red", "yellow", "white", "any"], "default": "any",
                   "description": "colour of the object (more reliable than 'any', which can confuse it with the white paper)"},
         "grasp_height_cm": {"type": "number", "default": 2.5,
                             "description": "fingertip height for the grip, about half the object's height (ball 4 cm -> 2.5)"},
         "align_wrist": {"type": "boolean", "default": False,
                         "description": "turn the wrist so the jaws close across a LONG object (box, bottle); not for balls"}}}},
    {"name": "place_at", "description": "Carry the held object to x/y cm (table frame), lower it until it almost "
     "touches, open the gripper, lift 5 cm. Use after a successful grasp.",
     "inputSchema": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                     "required": ["x", "y"]}},
    {"name": "run_skill", "description": "Run a LEARNED skill (a small neural network trained from human demos): fast and "
     "smooth (~20-40 s) but it only knows its one task and does not check its result. Skills: ball_to_paper = picks "
     "up the orange ball from anywhere on the mat and puts it on the white paper, then returns home. ALWAYS look at the "
     "returned images to check the result; if it failed, finish the task yourself with the other tools.",
     "inputSchema": {"type": "object", "properties": {"skill": {"type": "string", "enum": list(SKILLS)}}, "required": ["skill"]}},
    {"name": "go_home", "description": "Go back to the home (rest) pose: arm folded upright, all motors relaxed, the "
     "gripper camera looking forward across the whole table. Start and end every task here.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "open_dashboard", "description": "Open mission control in the browser: a live 3D twin of the arm (driven by "
     "its real joint readings), every object the cameras found (unknown ones named by Claude), both camera views with "
     "their detections, the step-by-step reasoning and a chat panel. Returns the link. simulation=true runs a "
     "clearly labelled demo without a robot.",
     "inputSchema": {"type": "object", "properties": {
         "simulation": {"type": "boolean", "description": "true = no robot needed, scripted demo run"}}}},
]


DASH_PORT = 8893


def _port_open(port):
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _get(path):
    with urllib.request.urlopen(f"http://127.0.0.1:{DASH_PORT}{path}", timeout=3) as r:
        return json.loads(r.read())


def open_dashboard(sim=False):
    """Start tools/vision_dashboard.py in the background (it keeps running after this call) and describe it."""
    started = False
    if not _port_open(DASH_PORT):
        log = arm_control.ROOT / "snapshots/dashboard.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(arm_control.ROOT / "tools/vision_dashboard.py")] + (["--sim"] if sim else [])
        subprocess.Popen(cmd, cwd=arm_control.ROOT, stdin=subprocess.DEVNULL, stdout=open(log, "a"),
                         stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(80):  # the arm's 3D meshes load in ~2 s
            if _port_open(DASH_PORT):
                break
            time.sleep(0.25)
        else:
            return f"FAILED: the dashboard did not start (see {log})"
        started = True
        time.sleep(1.5)
    try:
        robot, scene = _get("/api/robot"), _get("/api/scene")
    except Exception as e:
        return f"FAILED: something else is using port {DASH_PORT} ({e})"
    mode = "SIMULATION (scripted demo, no robot)" if robot["mode"] == "sim" else "LIVE"
    if sim and robot["mode"] != "sim":
        mode += " (it was already running live; stop it to switch to the simulation)"
    try:
        ip = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=3).stdout.split()[0]
    except (OSError, IndexError, subprocess.SubprocessError):
        ip = None
    objs = [o for o in scene.get("world", []) if o.get("known") or o.get("name")]
    seen = "; ".join(f"{o['label'] or o['name']} at ({o['x']:+.0f}, {o['y']:.0f}) cm" for o in objs) or "nothing yet"
    return (f"Mission control {'started' if started else 'is already running'}, {mode}.\n"
            f"Open: http://localhost:{DASH_PORT}" + (f"  (from Windows when this runs in WSL, or another device: "
                                                     f"http://{ip}:{DASH_PORT})" if ip else "") + "\n"
            "It shows the arm in 3D (drag to rotate, right-drag to pan, scroll to zoom), both cameras with what they "
            "detect, the task phases with timings, the reasoning stream and a chat panel ('Claude · MCP').\n"
            f"Arm: {'connected' if robot.get('ok') else 'NOT connected (is NormaCore Station running?)'}. "
            f"Objects: {seen}.")

ARM = None  # geometry loaded once; Station connection opened per call


async def call(name, args):
    global ARM
    if name == "open_dashboard":  # needs no robot connection (it can run as a simulation)
        text = open_dashboard(bool(args.get("simulation")))
        return {"content": [{"type": "text", "text": text}], "isError": text.startswith("FAILED")}
    if ARM is None:
        ARM = arm_control.Arm()
    await ARM.connect()
    with_images = True
    if name == "look":
        text = ARM.describe(await ARM.read())
        try:  # what the table camera recognises, with positions you can move to (arm base frame, cm)
            timg = await ARM.table_image()
            objs = vision.table_objects(timg)
            if not any(o["label"] == "orange ball" for o in objs):
                on, _n = vision.ball_on_top(timg, objs)
                if on is not None:
                    objs.append({"label": f"orange ball (on top of the {on['label']})", "x": on["x"], "y": on["y"]})
            text += ("\nTable camera detections (x, y in cm, ±2-3 cm; mugs ~10 cm tall, cube 5.7 cm): "
                     + ("; ".join(f"{o['label']} at ({o['x']:+.1f}, {o['y']:.1f})" for o in objs) or "none"))
        except Exception as e:
            text += f"\n(table camera detection failed: {e})"
    elif name == "get_state":
        text, with_images = ARM.describe(await ARM.read()), False
    elif name == "move_by":
        text = await ARM.move_tip(delta_cm=[float(args.get(k, 0)) for k in ("dx", "dy", "dz")], orient=args.get("orient", "keep"))
    elif name == "move_in_view":
        text = await ARM.move_in_view(float(args.get("right_cm", 0)), float(args.get("up_cm", 0)),
                                      float(args.get("down_cm", 0)), orient=args.get("orient", "keep"))
    elif name == "move_to":
        text = await ARM.move_tip(target_cm=[float(args[k]) for k in ("x", "y", "z")], orient=args.get("orient", "down"))
    elif name == "rotate_joint":
        text = await ARM.move_joint(args["joint"], float(args["degrees"]))
    elif name == "set_gripper":
        text = await ARM.set_gripper(float(args["open_percent"]))
    elif name == "ready":
        text = await ARM.ready()
    elif name == "put_ball_on_cube":
        text = await ARM.put_ball_on("rubiks cube", float(args.get("cube_height_cm", 5.7)))
    elif name == "put_ball_in_mug":
        text = await ARM.put_ball_in_mug(args.get("mug", "white mug"), float(args.get("mug_height_cm", 10.0)))
    elif name == "pick":
        text = await ARM.pick(args.get("color", "orange"), float(args.get("grasp_height_cm", 2.5)))
    elif name == "grasp":
        text = await ARM.grasp(args.get("color", "any"), float(args.get("grasp_height_cm", 2.5)),
                               align_wrist=bool(args.get("align_wrist", False)))
    elif name == "place_at":
        text = await ARM.place_at(float(args["x"]), float(args["y"]))
    elif name == "run_skill":
        text = run_skill(args["skill"])
        await ARM.connect()  # the skill ran in another process: fresh Station connection for the images
        text += "\n" + ARM.describe(await ARM.read())
    elif name == "go_home":
        text = await ARM.go_home()
    else:
        raise ValueError(f"unknown tool {name}")
    content = [{"type": "text", "text": text}]
    if with_images:
        for label, jpeg in await ARM.look():
            content.append({"type": "text", "text": label + ":"})
            content.append({"type": "image", "data": base64.b64encode(jpeg).decode(), "mimeType": "image/jpeg"})
    failed = text.startswith(("REFUSED", "STOPPED", "GRASP", "MISSED", "PLACE ABORTED", "FAILED", "DROP ABORTED")) \
        or "\nSTOPPED" in text
    failed = failed and not text.startswith("GRASPED")
    return {"content": content, "isError": failed}


def reply(msg_id, result=None, error=None):
    out = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        out["error"] = error
    else:
        out["result"] = result
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        method, msg_id, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
        if msg_id is None:  # notification
            continue
        try:
            if method == "initialize":
                reply(msg_id, {"protocolVersion": params.get("protocolVersion", "2025-06-18"),
                               "capabilities": {"tools": {}},
                               "serverInfo": {"name": "elrobot", "version": "0.1.0"},
                               "instructions": INSTRUCTIONS})
            elif method == "ping":
                reply(msg_id, {})
            elif method == "tools/list":
                reply(msg_id, {"tools": TOOLS})
            elif method == "tools/call":
                name, args = params["name"], params.get("arguments") or {}
                t_call = time.time()
                _log(f"Claude calls {name}({', '.join(f'{k}={v}' for k, v in args.items())})",
                     {"type": "tool_call", "tool": name, "args": args})
                try:
                    res = asyncio.run(call(name, args))
                    first = next((c.get("text", "") for c in res.get("content", []) if c.get("type") == "text"), "")
                    _log(f"{name} → {first.splitlines()[0] if first else 'done'}",
                         {"type": "tool_result", "tool": name, "ok": not res.get("isError"),
                          "seconds": round(time.time() - t_call, 1)})
                    reply(msg_id, res)
                except Exception as e:  # report to the model instead of crashing the server
                    traceback.print_exc(file=sys.stderr)
                    _log(f"{name} → ERROR: {e}", {"type": "tool_result", "tool": name, "ok": False,
                                                  "seconds": round(time.time() - t_call, 1)})
                    reply(msg_id, {"content": [{"type": "text", "text": f"ERROR: {e}"}], "isError": True})
            else:
                reply(msg_id, error={"code": -32601, "message": f"method not found: {method}"})
        except Exception as e:
            reply(msg_id, error={"code": -32603, "message": str(e)})


if __name__ == "__main__":
    main()
