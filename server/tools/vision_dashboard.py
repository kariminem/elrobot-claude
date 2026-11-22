"""ElRobot mission control: a live 3D digital twin of the arm (NormaCore URDF meshes driven by the real joint
readings), the objects the cameras found, both camera views with their detections, and the step-by-step reasoning
(including Claude's MCP tool calls) as it happens.

    ./tools/vision_dashboard          live (needs Station); open http://<wsl-ip>:8893 (printed at start)
    ./tools/vision_dashboard --sim    SIMULATION without a robot (scripted pick-and-drop, labelled as such everywhere)

Endpoints: /  (page)   /api/robot  /api/scene  /api/events?since=T  /api/stats  /api/meshes  /mesh/<link>.bin
           /live/table.jpg  /live/gripper.jpg  /img/<trace image>  POST /clear
Motion is recorded to snapshots/vision/states.jsonl (joint positions whenever the arm moves) for later replays.
"""
import argparse
import asyncio
import json
import logging
import os
import re
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arm_control  # noqa: E402
import arm_geometry as ag  # noqa: E402
import vision  # noqa: E402

PORT = 8893
ROOT = Path(__file__).resolve().parent.parent
WEB = Path(__file__).resolve().parent / "web"
CHAT_DIR = ROOT / "snapshots/chat"  # cwd of the headless Claude runs (chat + object naming) and the object crops
STARTED = time.time()
UNKNOWN = {"on": False}  # unknown-object detection + naming by Claude: off unless --unknown (false alarms cost quota)
NAMER = {"calls": 0, "last": 0.0, "dead": None}  # at most 6 namings, one per minute, none after any error
STATES = vision.TRACE_DIR / "states.jsonl"
LIVE = {"table": None, "gripper": None, "info": "starting…", "t": 0.0, "objects": [], "objects_t": 0.0,
        "robot": None, "robot_err": None, "mode": "live"}
LOCK = threading.Lock()
SIM_EVENTS = []  # simulation mode only: never written to the real trace
GRIPPER_JOINTS = ("rev_motor_08", "rev_motor_08_1", "rev_motor_08_2")
MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css", ".json": "application/json",
        ".jpg": "image/jpeg", ".png": "image/png", ".svg": "image/svg+xml"}


# ----------------------------------------------------------------------------------------------- arm model + FK
def load_meshes():
    """Every link's visual mesh as a triangle soup (float32 x,y,z in cm, link frame). arm_geometry keeps only a
    point cloud (enough for collision checks); drawing needs the triangles."""
    root = ET.parse(ag.URDF).getroot()
    out = {}
    for link in root.findall("link"):
        m = link.find("visual/geometry/mesh")
        if m is None:
            continue
        o = link.find("visual/origin")
        data = (ag.URDF.parent / m.get("filename")).read_bytes()
        n = struct.unpack_from("<I", data, 80)[0]
        if 84 + n * 50 == len(data):
            tri = np.frombuffer(data, dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]), count=n, offset=84)
            pts = tri["v"].reshape(-1, 3).astype(float)
        else:
            pts = np.array([[float(x) for x in line.split()[1:4]] for line in data.decode().splitlines()
                            if line.strip().startswith("vertex")])
        pts = pts * ag._vec(m.get("scale"), "1 1 1")
        t = ag._tf(ag._vec(o.get("xyz") if o is not None else None), ag._rpy(*ag._vec(o.get("rpy") if o is not None else None)))
        out[link.get("name")] = ((t[:3, :3] @ pts.T).T + t[:3, 3]).astype(np.float32) * 100.0
    return out


def full_transforms(geo, norm8):
    """Link poses (base frame, m) like arm_geometry.link_transforms, plus the gripper: gear angle and both jaws
    slid according to the gripper opening (0 = closed, 1 = fully open). In the URDF, joint position 0 is the OPEN
    gripper (jaw centroids 4.8 cm apart) and +-0.0255 m closes it (1.1 cm apart), hence (1 - opening)."""
    tfs = geo.link_transforms(norm8)
    c = 1.0 - float(np.clip(norm8[7], 0.0, 1.0))  # how far closed
    for name in GRIPPER_JOINTS:
        j = geo.joints[name]
        t = tfs[j["parent"]] @ ag._tf(j["xyz"], ag._rpy(*j["rpy"]))
        if j["type"] == "revolute":
            t = t @ ag._tf(rot=ag._axis_angle(j["axis"], j["lower"] + c * (j["upper"] - j["lower"])))
        else:
            a = j["axis"] / np.linalg.norm(j["axis"])
            t = t @ ag._tf(a * ((j["lower"] if j["lower"] < 0 else j["upper"]) * c))
        tfs[j["child"]] = t
    return tfs


HOLD_OPEN = 0.8  # in the URDF model the open jaws touch a 4.5 cm ball at ~0.8; the real gripper squeezes the soft
#                 ball to ~0.3. While holding, draw the jaws AT the ball's surface (the HUD keeps the real reading).


def robot_packet(arm, norm8, torque=None, errors=None, sim=False):
    shown = np.array(norm8, float)
    if (LIVE.get("story") or {}).get("held"):
        shown[7] = max(shown[7], HOLD_OPEN)
    tfs = full_transforms(arm.geo, shown)
    links = {}
    for name, t in tfs.items():
        m = t.copy()
        m[:3, 3] *= 100.0  # cm, like the meshes
        links[name] = [round(float(v), 5) for v in m.reshape(-1)]
    tip, axis = arm.fk(norm8)
    return {"t": time.time(), "sim": sim, "norm": [round(float(v), 4) for v in norm8],
            "deg": [round(v, 1) for v in arm.angles_deg(norm8)], "gripper": round(100 * float(norm8[7])),
            "torque": list(torque or []), "errors": [str(e) if e else "" for e in (errors or [])],
            "tip": [round(float(v) * 100, 2) for v in tip], "axis": [round(float(v), 4) for v in axis], "links": links}


# ----------------------------------------------------------------------------------------------- live loops
def _objects_json(objs):
    out = []
    for o in objs:
        d = {"label": o["label"], "x": round(float(o["x"]), 1), "y": round(float(o["y"]), 1)}
        if "handle_px" in o:
            d["handle_px"] = int(o["handle_px"])
        if o.get("on"):
            d["on"], d["z"] = o["on"], float(o["z"])
        out.append(d)
    return out


# ----------------------------------------------------------------------------------------------- world model
class World:
    """The dashboard's memory of the scene, which the 3D view draws. One entry per object:
    - table-camera positions jitter 1-2 cm from frame to frame: the entry only moves when the object really moved
      (3 consecutive readings agree with each other and are > 2.5 cm away) or the readings settle > 0.8 cm off;
    - a gripper-camera measurement (the precise one) is kept until the table camera sees the object moved > 3 cm;
    - unknown things on the mat get an entry too, and Claude names them from a camera crop."""
    KEEP_KNOWN_S = 45  # a known object unseen this long is gone (the ball is kept: it hides in the gripper / a mug)

    def __init__(self):
        self.lock = threading.Lock()
        self.objs = {}
        self.n_unknown = 0

    @staticmethod
    def _moved(hist, ref, far_cm):
        tail = np.array(hist)[-3:]
        if len(tail) == 3 and np.all(np.linalg.norm(tail - ref, axis=1) > far_cm) and np.ptp(tail, axis=0).max() < 2.0:
            return np.median(tail, axis=0)
        return None

    def _track(self, o, x, y):
        o["hist"].append((x, y))
        here = np.array([o["x"], o["y"]])
        if o["src"] == "gripper":  # measured precisely: only a real move (seen by the table camera) replaces it
            m = self._moved(o["hist"], np.array(o["table_ref"]), 3.0)
            if m is not None:
                o.update(src="table", x=float(m[0]), y=float(m[1]), moved_t=time.time())
            return
        m = self._moved(o["hist"], here, 2.5)
        if m is not None:
            o.update(x=float(m[0]), y=float(m[1]), moved_t=time.time())
            o["hist"] = deque([(o["x"], o["y"])], maxlen=12)
            return
        last = np.array(o["hist"])[-8:]
        med = np.median(last, axis=0)
        if len(last) >= 6 and np.linalg.norm(med - here) > 0.8 and (np.linalg.norm(last - here, axis=1) > 0.8).sum() >= 6:
            o["x"], o["y"] = float(med[0]), float(med[1])

    def see_known(self, objs, t):
        with self.lock:
            for d in objs:
                lab = d["label"]
                o = self.objs.get(lab)
                if o is None:
                    self.objs[lab] = {"id": lab, "label": lab, "known": True, "x": float(d["x"]), "y": float(d["y"]),
                                      "z": float(d.get("z", 0.0)), "on": d.get("on"), "src": "table", "seen": t, "first": t,
                                      "hist": deque([(float(d["x"]), float(d["y"]))], maxlen=12)}
                    continue
                o["seen"], o["z"], o["on"] = t, float(d.get("z", 0.0)), d.get("on")
                self._track(o, float(d["x"]), float(d["y"]))
            for k in [k for k, o in self.objs.items() if o["known"] and k != "orange ball" and t - o["seen"] > self.KEEP_KNOWN_S]:
                del self.objs[k]

    def measured(self, label, table_xy, xy):
        with self.lock:
            o = self.objs.get(label)
            if o is None:
                o = self.objs[label] = {"id": label, "label": label, "known": True, "z": 0.0, "on": None, "seen": time.time(),
                                        "first": time.time(), "hist": deque(maxlen=12)}
            ref = np.median(np.array(o["hist"])[-8:], axis=0).tolist() if len(o["hist"]) else list(table_xy)
            o.update(src="gripper", x=float(xy[0]), y=float(xy[1]), table_ref=ref, table_said=list(table_xy))

    def see_unknowns(self, blobs, t, rgb):
        with self.lock:
            for b in blobs:
                cands = [o for o in self.objs.values() if not o["known"]]
                o = min(cands, key=lambda o: np.hypot(o["x"] - b["x"], o["y"] - b["y"]), default=None)
                if o is not None and np.hypot(o["x"] - b["x"], o["y"] - b["y"]) < 4.0:
                    o["seen"], o["n"] = t, o["n"] + 1
                    o["w_cm"], o["d_cm"] = 0.8 * o["w_cm"] + 0.2 * b["w_cm"], 0.8 * o["d_cm"] + 0.2 * b["d_cm"]
                    self._track(o, b["x"], b["y"])
                    if o["name"] is None and o["n"] % 5 == 0:
                        o["crop"] = _context(rgb, b["box"])
                    continue
                self.n_unknown += 1
                self.objs[f"u{self.n_unknown}"] = {
                    "id": f"u{self.n_unknown}", "label": None, "known": False, "x": b["x"], "y": b["y"], "z": 0.0, "on": None,
                    "src": "table", "seen": t, "first": t, "n": 1, "w_cm": b["w_cm"], "d_cm": b["d_cm"], "name": None,
                    "naming": False, "crop": _context(rgb, b["box"]), "hist": deque([(b["x"], b["y"])], maxlen=12)}
            # called only while the arm is out of the camera's way: an unknown not seen for 8 s has been taken away
            for k in [k for k, o in self.objs.items() if not o["known"] and o["src"] != "claude" and t - o["seen"] > 8]:
                del self.objs[k]

    def set_claude_objects(self, items):
        """Replace the objects from the last 'Ask Claude' scan (they stay until the next scan)."""
        with self.lock:
            for k in [k for k, o in self.objs.items() if o["src"] == "claude"]:
                del self.objs[k]
            t = time.time()
            for i, it in enumerate(items):
                self.objs[f"c{i}"] = {"id": f"c{i}", "label": None, "known": False, "x": it["x"], "y": it["y"], "z": 0.0,
                                      "on": None, "src": "claude", "seen": t, "first": t, "n": 99, "w_cm": it["w_cm"],
                                      "d_cm": it["d_cm"], "name": it["name"], "naming": False, "crop": None,
                                      "hist": deque([(it["x"], it["y"])], maxlen=12)}

    def snapshot(self):
        now = time.time()
        with self.lock:
            out = []
            for o in self.objs.values():
                d = {k: o.get(k) for k in ("id", "label", "known", "src", "on", "name", "naming", "by_user")}
                d.update(x=round(o["x"], 1), y=round(o["y"], 1), z=round(o.get("z") or 0.0, 1), age=round(now - o["seen"], 1))
                if o["src"] == "gripper":
                    d["table_said"] = [round(v, 1) for v in o["table_said"]]
                if not o["known"]:
                    d.update(w_cm=round(o["w_cm"], 1), d_cm=round(o["d_cm"], 1),
                             confirmed=o["src"] == "claude" or (o["n"] >= 5 and now - o["first"] > 2.0))
                out.append(d)
            return out


def _context(rgb, box, margin=14):
    """What Claude looks at to name a thing: the whole table view with a RED box around it (scale and context: the
    mugs, the arm) next to a zoom on it. The table camera is only 320x240, so names are best guesses."""
    x, y, w, h = box
    H, W = rgb.shape[:2]
    big = cv2.resize(rgb, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(big, (3 * x - 9, 3 * y - 9), (3 * (x + w) + 9, 3 * (y + h) + 9), (255, 0, 0), 3)
    c = rgb[max(0, y - margin):min(H, y + h + margin), max(0, x - margin):min(W, x + w + margin)]
    c = cv2.resize(c, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
    c = cv2.resize(c, None, fx=big.shape[0] / c.shape[0], fy=big.shape[0] / c.shape[0])
    return np.hstack([big, c])


WORLD = World()


# ----------------------------------------------------------------------------------------------- Claude (headless)
def _claude(args, timeout):
    """Run Claude Code headless (the user's own login, no API key) in CHAT_DIR."""
    CHAT_DIR.mkdir(parents=True, exist_ok=True)
    return subprocess.run(["claude", "-p", *args], cwd=CHAT_DIR, capture_output=True, text=True, timeout=timeout)


def _emit(kind, text, img=None, data=None):
    """An event for the reasoning stream: the real trace when live, memory only in simulation."""
    if LIVE["mode"] == "sim":
        e = {"t": time.time(), "kind": kind, "text": text}
        if data:
            e["data"] = data
        SIM_EVENTS.append(e)
    else:
        vision.trace(kind, text, img, data=data)


def namer_loop():
    """Claude looks at each new (confirmed) unknown object once and names it."""
    while True:
        time.sleep(1.0)
        if NAMER["dead"] or NAMER["calls"] >= 6 or time.time() - NAMER["last"] < 60:
            continue
        with WORLD.lock:
            o = next((o for o in WORLD.objs.values() if not o["known"] and o["name"] is None and not o["naming"]
                      and o["n"] >= 5 and time.time() - o["first"] > 2.0 and o.get("crop") is not None), None)
            if o is None:
                continue
            o["naming"] = True
            crop, oid, xy = o["crop"], o["id"], (o["x"], o["y"])
        f = CHAT_DIR / f"object_{oid}_{int(time.time())}.jpg"
        CHAT_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(f), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        name = None
        NAMER["calls"] += 1
        NAMER["last"] = time.time()
        try:
            r = _claude([f"Read the image {f.name}. Left: a fixed low-resolution camera's view of a robot's black work mat "
                         "on a desk (the robot arm stands at the back). Right: a zoom on the thing inside the RED box. The "
                         "robot does not know that thing. What is it? Reply with ONLY its name, 1-3 lowercase words.",
                         "--model", "sonnet", "--tools", "Read", "--allowedTools", "Read",
                         "--output-format", "json"], timeout=90)
            res = json.loads(r.stdout)
            name = (res.get("result") or "").strip().strip(".'\"").lower()
            # an error or a usage-limit message must never become a "name": stop naming altogether
            if res.get("is_error") or res.get("subtype") != "success" or not re.fullmatch(r"[a-z][a-z' -]{1,30}", name):
                NAMER["dead"] = (res.get("result") or "unexpected reply")[:120]
                name = None
                logging.warning("object naming stopped: %s", NAMER["dead"])
        except Exception as e:  # stop naming: never loop on failures (each try costs quota)
            NAMER["dead"] = str(e)[:120]
            logging.warning("object naming stopped: %s", e)
        with WORLD.lock:
            o = WORLD.objs.get(oid)
            if o is not None:
                o["naming"] = False
                o["name"] = name or o["name"]
        if name:
            _emit("claude", f"Claude looked at a new object on the mat at ({xy[0]:+.0f}, {xy[1]:.0f}) cm: probably a {name}",
                  cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), {"type": "named", "id": oid, "name": name})


def mcp_config():
    """The MCP config the chat's Claude uses: the project's .mcp.json, or (inside the published package, which has
    none) one that starts the elrobot server with this same Python."""
    f = ROOT / ".mcp.json"
    if f.exists():
        return f
    CHAT_DIR.mkdir(parents=True, exist_ok=True)
    g = CHAT_DIR / "mcp.json"
    g.write_text(json.dumps({"mcpServers": {"elrobot": {"command": sys.executable,
                                                         "args": [str(ROOT / "tools/elrobot_mcp.py")]}}}, indent=2))
    return g


SCAN = {"busy": False, "error": None, "t": 0.0, "found": []}
KNOWN_NAMES = ("orange ball", "white mug", "black mug", "rubiks cube", "rubik's cube")


def scan_table():
    """'Ask Claude what's here': Claude looks at the table camera once and lists everything on the mat (also what the
    colour rules can't see, like a clear glass mug). Items on top of / next to a tracked object are parts of it."""
    try:
        rgb = LIVE.get("table_rgb")
        if rgb is None:
            raise RuntimeError("no table camera frame yet")
        CHAT_DIR.mkdir(parents=True, exist_ok=True)
        f = CHAT_DIR / "table_scan.jpg"
        big = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(f), big)
        H, W = big.shape[:2]
        r = _claude([f"Read the image {f.name} ({W}x{H} px): a fixed camera looking at a robot arm's black work mat. "
                     "List every object lying or standing ON the mat. Ignore the arm, the wall, cables and anything off "
                     "the mat. A mug's handle is part of the mug: never list a handle on its own. Reply with ONLY a JSON "
                     'array, one item per object: {"name": short lowercase name, "box": [x0, y0, x1, y1] in pixels of '
                     'this image, "base": [x, y] pixel where the object touches the mat}.',
                     "--model", "sonnet", "--tools", "Read", "--allowedTools", "Read", "--output-format", "json"], timeout=120)
        res = json.loads(r.stdout)
        if res.get("is_error") or res.get("subtype") != "success":
            raise RuntimeError((res.get("result") or "Claude did not answer")[:160])
        txt = res.get("result") or ""
        items = json.loads(txt[txt.index("["):txt.rindex("]") + 1])
        known = [o for o in WORLD.snapshot() if o["known"]]
        out = []
        for it in items:
            name = str(it.get("name", "")).strip().lower()[:40]
            (x0, y0, x1, y1), (bu, bv) = it["box"], it["base"]
            x, y = (float(c) for c in vision.px_to_cm(bu / 2, bv / 2))
            if not name or any(k in name for k in KNOWN_NAMES):
                continue  # the colour detectors already track these, precisely
            if any(np.hypot(x - o["x"], y - o["y"]) < (7.0 if "mug" in o["label"] else 4.0) for o in known):
                continue  # a part of a tracked object (a handle, a sticker)
            w_cm = float(np.hypot(*np.subtract(vision.px_to_cm(x0 / 2, y1 / 2), vision.px_to_cm(x1 / 2, y1 / 2))))
            d_cm = float(np.hypot(*np.subtract(vision.px_to_cm(bu / 2, y0 / 2), vision.px_to_cm(bu / 2, y1 / 2))))
            out.append({"name": name, "x": x, "y": y, "w_cm": min(max(w_cm, 2.0), 25.0), "d_cm": min(max(d_cm, 2.0), 25.0)})
        WORLD.set_claude_objects(out)
        SCAN.update(found=[o["name"] for o in out], error=None)
        _emit("claude", "Claude looked at the table camera: " + (", ".join(
            f"{o['name']} at ({o['x']:+.0f}, {o['y']:.0f}) cm" for o in out) or "nothing beyond the tracked objects"),
            big, {"type": "scan", "found": out})
    except Exception as e:
        SCAN["error"] = str(e)[:200]
        logging.warning("scan failed: %s", e)
    finally:
        SCAN.update(busy=False, t=time.time())


AUTO = {"on": True, "calls": 0, "max": 30, "cooldown": 30.0, "last": 0.0, "status": "starting", "change": 0.0}


def _thumb(rgb):
    """Small, smoothed, brightness-normalised grey view of the mat: compares scenes, not light levels."""
    g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(rgb, (100, 75), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
                         .astype(np.float32), (5, 5), 0)
    m = np.zeros((75, 100), np.uint8)
    cv2.fillPoly(m, [(vision.MAT_ROI * np.array([100 / 320, 75 / 240])).astype(np.int32)], 1)
    m = cv2.erode(m, np.ones((3, 3), np.uint8)) > 0
    vals = g[m]
    return (g - vals.mean()) / (vals.std() + 1e-3), m


def auto_scan_loop():
    """Claude looks at the table by itself when the scene changed: something new appears or goes, the scene has been
    still for 2.5 s (no hands moving) and the arm is home. Cheap image comparison decides; Claude is only called then
    (at most one scan per 30 s, 30 per session, none after an error)."""
    ref = prev = None
    still_since = None
    while True:
        time.sleep(0.5)
        try:
            rgb, r, st = LIVE.get("table_rgb"), LIVE.get("robot"), LIVE.get("story") or {}
            if not AUTO["on"] or rgb is None or SCAN["busy"]:
                continue
            arm_home = LIVE["mode"] == "sim" or (r is not None and r["tip"][1] < 13.0)
            active = time.time() - LIVE.get("last_event_t", 0.0) < 10.0  # a task / chat step happened just now
            if not arm_home or active:
                AUTO["status"], prev, still_since = "paused while the arm works", None, None
                continue
            t, m = _thumb(rgb)
            if prev is not None and float(np.abs(t - prev)[m].mean()) > 0.15:
                still_since = None  # something is moving (hands)
            elif still_since is None:
                still_since = time.time()
            prev = t
            if still_since is None or time.time() - still_since < 2.5:
                AUTO["status"] = "scene moving, waiting until it is still"
                continue
            AUTO["change"] = round(float((np.abs(t - ref)[m] > 0.8).mean()), 4) if ref is not None else 1.0
            if AUTO["change"] < 0.004:
                AUTO["status"] = "watching the table"
                continue
            if SCAN["error"] or AUTO["calls"] >= AUTO["max"]:
                AUTO["status"] = "stopped: " + (SCAN["error"] or f"{AUTO['max']} scans used")
                continue
            if time.time() - AUTO["last"] < AUTO["cooldown"]:
                AUTO["status"] = "something changed: next look in a few seconds"
                continue
            AUTO.update(calls=AUTO["calls"] + 1, last=time.time(), status="something changed: Claude is looking")
            SCAN.update(busy=True, error=None)
            scan_table()
            ref = t
        except Exception as e:  # keep watching
            AUTO["status"] = f"watch error: {e}"


CHAT = {"msgs": [], "busy": False, "session": None, "model": "sonnet", "proc": None, "n": 0}
CHAT_LOCK = threading.Lock()
CHAT_PROMPT = (
    "You are Claude, chatting with a person through the robot's live dashboard while controlling a REAL robot arm "
    "(NormaCore ElRobot) with the elrobot MCP tools. They watch the arm, both cameras and a live 3D twin. Do what they "
    "ask, one step at a time, and check the camera images in every tool result. Replies: 1-2 short sentences, plain "
    "text. 'Go to X' / 'go over X' without picking: call look to get X's table-camera position, then move_to the "
    "fingertips 8-10 cm above it (above a mug's rim: ~20 cm) and stop; never touch it. 'Pick X': pick. 'Put the ball in "
    "the white mug': put_ball_in_mug. 'Put the ball on the cube': put_ball_on_cube. 'Home': go_home. Never claim a "
    "success the images don't show. If a tool fails, say why in plain words.")


def chat_add(role, text, **kw):
    with CHAT_LOCK:
        CHAT["n"] += 1
        CHAT["msgs"].append({"i": CHAT["n"], "t": time.time(), "role": role, "text": text, **kw})
        del CHAT["msgs"][:-300]


def chat_send(text):
    with CHAT_LOCK:
        if CHAT["busy"]:
            return "Claude is still working on the last message (press Stop to interrupt)."
        CHAT["busy"] = True
    chat_add("user", text)
    cmd = ["claude", "-p", text, "--model", CHAT["model"], "--mcp-config", str(mcp_config()), "--strict-mcp-config",
           "--tools", "", "--allowedTools", "mcp__elrobot__*", "--output-format", "stream-json", "--verbose",
           "--append-system-prompt", CHAT_PROMPT]
    if CHAT["session"]:
        cmd += ["--resume", CHAT["session"]]
    CHAT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        p = subprocess.Popen(cmd, cwd=CHAT_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
                             start_new_session=True)
    except OSError as e:
        CHAT["busy"] = False
        chat_add("error", f"could not start Claude: {e}")
        return None
    CHAT["proc"] = p
    threading.Thread(target=chat_reader, args=(p,), daemon=True).start()
    return None


def _result_text(c):
    content = c.get("content")
    if isinstance(content, str):
        return content
    return next((x.get("text", "") for x in (content or []) if x.get("type") == "text"), "")


def chat_reader(p):
    got_result = False
    for line in p.stdout:
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        typ = e.get("type")
        if typ == "system" and e.get("subtype") == "init":
            CHAT["session"] = e.get("session_id") or CHAT["session"]
        elif typ == "assistant":
            for c in e.get("message", {}).get("content", []):
                if c.get("type") == "text" and c.get("text", "").strip():
                    chat_add("assistant", c["text"].strip())
                elif c.get("type") == "tool_use":
                    chat_add("tool", c.get("name", "").replace("mcp__elrobot__", ""), input=c.get("input") or {})
        elif typ == "user":
            for c in e.get("message", {}).get("content", []) if isinstance(e.get("message", {}).get("content"), list) else []:
                if c.get("type") == "tool_result":
                    txt = _result_text(c).strip()
                    chat_add("tool_result", txt.splitlines()[0][:220] if txt else "done", error=bool(c.get("is_error")))
        elif typ == "result":
            got_result = True
            CHAT["session"] = e.get("session_id") or CHAT["session"]
            if e.get("is_error"):
                chat_add("error", str(e.get("result") or e.get("subtype"))[:400])
            chat_add("done", f"{(e.get('duration_ms') or 0) / 1000:.1f} s")
    p.wait()
    if not got_result and p.returncode not in (0, None, -signal.SIGTERM):
        chat_add("error", (p.stderr.read() or f"Claude exited ({p.returncode})")[-400:])
    with CHAT_LOCK:
        CHAT["busy"], CHAT["proc"] = False, None


def _descendants(pid):
    """Every process started (directly or not) by pid: the chat's Claude starts its MCP server as a child process."""
    kids = {}
    for d in Path("/proc").glob("[0-9]*"):
        try:
            kids.setdefault(int((d / "stat").read_text().rsplit(")", 1)[1].split()[1]), []).append(int(d.name))
        except (OSError, ValueError, IndexError):
            pass
    out, todo = [], [pid]
    while todo:
        for c in kids.get(todo.pop(), []):
            out.append(c)
            todo.append(c)
    return out


def _freeze_arm():
    """Goal = where every joint is right now: the arm stops moving and holds (the gripper keeps its grip)."""
    async def go():
        arm = arm_control.Arm()
        await arm.connect()
        await arm.hold()
    try:
        asyncio.run(go())
        return True
    except Exception as e:
        logging.warning("freeze failed: %s", e)
        return False


def chat_stop():
    """STOP: kill the chat's Claude and everything it started (its MCP server, mid-move), any ./tools/arm task, then
    freeze the arm where it is. Leaves other Claude sessions' MCP servers alone."""
    pids = []
    p = CHAT.get("proc")
    if p is not None and p.poll() is None:
        pids += _descendants(p.pid) + [p.pid]
    for d in Path("/proc").glob("[0-9]*"):  # tasks started from the terminal (./tools/arm mug ...)
        try:
            if "tools/arm_control.py" in (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace"):
                pids.append(int(d.name))
        except OSError:
            pass
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    frozen = _freeze_arm()
    _emit("error", "STOP pressed: " + ("arm frozen where it is" if frozen else "could NOT reach the arm to freeze it")
          + (f", {len(pids)} running process(es) stopped" if pids else ""))
    chat_add("system", "■ STOP: " + ("the arm is frozen where it is" if frozen else "could not reach the arm: use the power switch")
             + (" and Claude was stopped" if p is not None and pids else ""))
    return frozen


def camera_loop():
    async def run():
        arm = arm_control.Arm()
        while True:
            try:
                await arm.connect()
                while True:
                    imgs = await arm.camera_images()
                    out, objs = {}, []
                    if "table" in imgs:
                        objs = vision.table_objects(imgs["table"])
                        if not any(o["label"] == "orange ball" for o in objs):  # resting ON something?
                            on, _n = vision.ball_on_top(imgs["table"], objs)
                            if on is not None:
                                objs = objs + [{"label": "orange ball", "x": on["x"], "y": on["y"], "on": on["label"],
                                                "z": vision.SURFACE_HEIGHT_CM[on["label"]], "box": on["box"]}]
                        now = time.time()
                        LIVE["table_rgb"] = imgs["table"]
                        WORLD.see_known(objs, now)
                        r = LIVE.get("robot")
                        clear_view = UNKNOWN["on"] and r is not None and r["tip"][1] < 13.0  # folded back, not over the mat
                        unknown = vision.unknown_objects(imgs["table"], objs) if clear_view else []
                        if clear_view:
                            WORLD.see_unknowns(unknown, now, imgs["table"])
                        out["table"] = vision.annotate(imgs["table"], objs + [dict(u, label="?") for u in unknown], outline=True,
                                                       title="TABLE camera (green = mat, the only area used): "
                                                       + (", ".join(o["label"] for o in objs) or "no objects"))
                    st = await arm.read()
                    if "gripper" in imgs:
                        g = imgs["gripper"]
                        down = -arm.fk(st.norm())[1][2] > 0.7  # pointing > ~45 deg down: the view is all mat
                        gobjs = []
                        active = vision.get_active() or "orange ball"
                        if down:  # only what the robot is working on right now (no guessing at everything)
                            kind = {"orange ball": "orange", "white mug": "white", "black mug": "dark"}.get(active, "orange")
                            b = vision.gripper_blob(g, kind)
                            if b is not None:
                                gobjs.append({**b, "label": active})
                        out["gripper"] = vision.annotate(
                            g, gobjs, target=(160, 145) if down else None,
                            title="GRIPPER camera, looking for the " + active + ": " + ((", ".join(o["label"] for o in gobjs) or "not in view") if down
                                                                                      else "looking forward (detection runs when pointing down)"))
                    info = arm.describe(st).splitlines()
                    with LOCK:
                        for k, im in out.items():
                            LIVE[k] = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                        if "table" in imgs:
                            LIVE["objects"], LIVE["objects_t"] = _objects_json(objs), time.time()
                        LIVE["info"] = " | ".join(info[:3])
                        LIVE["t"] = time.time()
                    await asyncio.sleep(0.2)
            except Exception as e:
                with LOCK:
                    LIVE["info"] = f"Station not reachable ({type(e).__name__}: {e}); retrying…"
                await asyncio.sleep(2)
    asyncio.run(run())


def state_loop():
    """Joint readings at ~10 Hz for the 3D twin (own Station connection), recorded whenever the arm moves."""
    async def run():
        arm = arm_control.Arm()
        last_q, last_rec = None, 0.0
        while True:
            try:
                await arm.connect()
                while True:
                    st = await arm.read()
                    n = st.norm()
                    pkt = robot_packet(arm, n, st.torque, st.errors)
                    with LOCK:
                        LIVE["robot"], LIVE["robot_err"] = pkt, None
                    now = time.time()
                    if last_q is None or max(abs(a - b) for a, b in zip(n, last_q)) > 0.002 or now - last_rec > 10:
                        with open(STATES, "a") as f:
                            f.write(json.dumps({"t": round(now, 3), "q": pkt["norm"]}) + "\n")
                        last_q, last_rec = n, now
                    await asyncio.sleep(0.1)
            except Exception as e:
                with LOCK:
                    LIVE["robot_err"] = f"{type(e).__name__}: {e}"
                await asyncio.sleep(2)
    asyncio.run(run())


# ----------------------------------------------------------------------------------------------- simulation
def sim_loop():
    """SIMULATION: a scripted ball -> white mug run on the 3D twin, for testing the page without a robot.
    Positions from the last real scene (2026-11-21); events go to memory only, every text starts with [SIM]."""
    arm = arm_control.Arm()
    home = np.array(json.loads(arm_control.HOME_FILE.read_text())["position_norm"], float)
    ball, mug_tab, mug = (-12.3, 27.4), (15.6, 30.4), (11.8, 27.8)
    down = np.array([0.0, 0.0, -1.0])

    def ik(p_cm, seed, orient=True):
        q, err = arm.ik(seed, np.array(p_cm) / 100.0, down if orient else None)
        if err > 0.01 and orient:
            q, err = arm.ik(seed, np.array(p_cm) / 100.0, None)
        return np.array(q, float)

    def g(q, o):
        q = q.copy()
        q[7] = o
        return q

    k_survey = g(ik([mug[0] - 2.3, mug[1] - 4.8, 17.4], home, orient=False), 0.98)
    k_above = g(ik([ball[0], ball[1] - 2, 10], k_survey), 1.0)
    k_at = g(ik([ball[0], ball[1], 2.8], k_above), 1.0)
    k_lift = g(ik([ball[0], ball[1], 9.0], k_at), 0.3)
    k_transit = g(ik([mug[0], mug[1], 16.5], k_lift, orient=False), 0.3)
    k_drop = g(ik([mug[0], mug[1], 14.0], k_transit, orient=False), 0.3)

    script = [  # (pose, seconds to get there, events when it starts)
        (g(home, 0.8), 2.0, [("task", "TASK: put the orange ball in the white mug", {"type": "task_start", "task": "ball → white mug"}),
                             ("detect", "table camera: white mug at (+16, 30) cm, black mug at (-2, 41) cm, rubiks cube at (-31, 37) cm, orange ball at (-12, 27) cm", None)]),
        (k_survey, 3.0, [("think", "white mug at about (+15.6, 30.4) cm (table camera, ±2.5 cm, worse at the edges): hover 16 cm above it with the EMPTY gripper and let the gripper camera find the rim's centre", None)]),
        (k_survey, 1.5, [("result", "white mug MEASURED: gripper axis meets the rim at (+11.8, 27.8) cm (table camera said (+15.6, 30.4))",
                          {"type": "measured", "label": "white mug", "table": list(mug_tab), "measured": list(mug)})]),
        (k_above, 3.0, [("think", "target = orange ball at (-12.3, 27.4) cm → fly 10 cm above the table, then let the gripper camera guide the grasp",
                         {"type": "target", "label": "orange ball", "xy": list(ball), "on": None}),
                        ("gripper", "OPEN the gripper fully before going down, so the jaws pass around the object", None)]),
        (k_at, 3.0, [("move", "orange ball is +3 px right / -7 px down of the jaws' target: centred → go down", None)]),
        (g(k_at, 0.3), 1.5, [("gripper", "CLOSE: the orange ball is centred and the fingertips are at grasp height", None)]),
        (k_lift, 1.5, [("result", "GRASPED: the jaws stopped at 30% open (blocked by the object = holding it); lifted 6 cm",
                        {"type": "grasped", "label": "orange ball", "tip": [ball[0], ball[1], 2.8]})]),
        (k_transit, 3.0, [("think", "carry at 16.5 cm (ball 4 cm above any mug rim: the black mug is on the way), then lower to 14 cm over the white mug", None)]),
        (k_drop, 1.5, []),
        (g(k_drop, 1.0), 1.0, [("gripper", "OPEN over the white mug: fingertips 0.2 cm from the measured centre, 14 cm up",
                                {"type": "release", "label": "white mug", "tip": [mug[0], mug[1], 14.0], "target": list(mug)})]),
        (g(k_survey, 1.0), 2.5, [("detect", "gripper camera, back at the survey pose: the ball's centre is INSIDE the white mug's rim",
                                  {"type": "check", "label": "white mug", "verdict": "in"})]),
        (g(home, 1.0), 3.0, [("result", "SUCCESS: the gripper camera sees the ball inside the white mug's rim", None),
                             ("task", "done in 26 s: SUCCESS: the ball is in the white mug", {"type": "task_end", "verdict": "SUCCESS", "seconds": 26.0})]),
        (g(home, 1.0), 5.0, []),
    ]
    q = g(home, 0.8)
    while True:
        for target, secs, evs in script:
            for kind, text, data in evs:
                e = {"t": time.time(), "kind": kind, "text": "[SIM] " + text}
                if data:
                    e["data"] = data
                SIM_EVENTS.append(e)
            q0, t0 = q.copy(), time.time()
            while (a := (time.time() - t0) / secs) < 1.0:
                s = a * a * (3 - 2 * a)
                cur = q0 + (target - q0) * s
                with LOCK:
                    LIVE["robot"] = robot_packet(arm, cur, [1] * 8, [None] * 8, sim=True)
                time.sleep(0.05)
            q = target.copy()
        del SIM_EVENTS[:-60]


# ----------------------------------------------------------------------------------------------- events, story, stats
def read_events(n=400, since=None):
    if LIVE["mode"] == "sim":
        evs = list(SIM_EVENTS)
    else:
        f = vision.TRACE_DIR / "events.jsonl"
        evs = []
        if f.exists():
            for line in f.read_text().splitlines()[-n:]:
                try:
                    evs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return [e for e in evs if since is None or e["t"] > since]


PHASES = ["locate", "measure", "pick", "carry", "drop", "verify", "done"]


def story(evs):
    """The current (latest) task, told from its events: phase, timings and the facts the 3D view draws."""
    start = max((i for i, e in enumerate(evs) if (e.get("data") or {}).get("type") == "task_start"
                 or (e["kind"] == "task" and "TASK:" in e["text"])), default=None)
    st = {"task": None, "phase": "idle", "t0": None, "t_end": None, "verdict": None, "seconds": None,
          "phase_t": {}, "target": None, "measured": None, "grasped": None, "release": None, "check": None, "held": None}
    if start is None:
        return st
    for e in evs[start:]:
        d, txt = e.get("data") or {}, e["text"]
        typ = d.get("type")

        def phase(p):  # phases only move forward
            if st["phase"] not in PHASES or PHASES.index(p) >= PHASES.index(st["phase"]):
                st["phase"] = p
                st["phase_t"].setdefault(p, e["t"])
        if typ == "task_start" or (e["kind"] == "task" and "TASK:" in txt):
            st.update(task=d.get("task") or txt.split("TASK:")[-1].strip(), t0=e["t"])
            phase("locate")
        elif typ == "measured" or "EMPTY gripper" in txt:
            if typ == "measured":
                st["measured"] = d
            phase("measure")
        elif typ == "target":
            st["target"] = d
            phase("pick")
        elif e["kind"] == "gripper" and re.match(r"(\[SIM\] )?CLOSE", txt):
            st["held"] = (st["target"] or {}).get("label") or "orange ball"  # the jaws close on it: it moves with them
        elif typ == "grasped" or txt.startswith(("GRASPED", "[SIM] GRASPED")):
            st["grasped"], st["held"] = (d or {"label": "orange ball"}), "orange ball"
            phase("carry")
        elif re.match(r"(\[SIM\] )?MISSED: the jaws closed", txt):
            st["held"] = None
        elif typ == "release" or "OPEN over the" in txt:
            st["release"], st["held"] = d or None, None
            phase("drop")
        elif typ == "check" or "back at the survey pose" in txt:
            st["check"] = d or None
            phase("verify")
        elif typ == "task_end" or (e["kind"] == "task" and re.search(r"done in \d+ s", txt)):
            m = re.search(r"done in (\d+) s: (\w+)", txt)
            st["verdict"] = d.get("verdict") or (m.group(2) if m else None)
            st["seconds"] = d.get("seconds") or (float(m.group(1)) if m else None)
            st["t_end"] = e["t"]
            phase("done")
        elif e["kind"] == "result" and re.match(r"(\[SIM\] )?(FAILED|MISSED)", txt):
            st["verdict"] = st["verdict"] or "MISSED"
    return st


def stats():
    """Every ball task ever traced (current + rotated logs): verdicts and timings."""
    runs = []
    files = sorted(vision.TRACE_DIR.glob("events_*.jsonl")) + [vision.TRACE_DIR / "events.jsonl"]
    cur = None
    for f in files:
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            txt = e.get("text", "")
            if e.get("kind") == "task" and "TASK:" in txt:
                if cur:
                    runs.append({**cur, "verdict": "ABORTED", "seconds": None})
                cur = {"t": e["t"], "task": txt.split("TASK:")[-1].strip()}
            elif e.get("kind") == "task" and cur and (m := re.search(r"done in (\d+) s: (\w+)", txt)):
                runs.append({**cur, "verdict": m.group(2), "seconds": float(m.group(1))})
                cur = None
    if cur:
        runs.append({**cur, "verdict": "RUNNING" if time.time() - cur["t"] < 120 else "ABORTED", "seconds": None})
    try:  # verdicts the automatic check got wrong, corrected after looking at the images (with the reason)
        fixes = json.loads((vision.TRACE_DIR / "corrections.json").read_text())
    except (OSError, json.JSONDecodeError):
        fixes = {}
    for r in runs:
        fx = fixes.get(f"{r['t']:.1f}")
        if fx:
            r.update(auto=r["verdict"], verdict=fx["verdict"], why=fx["why"])
    today = time.strftime("%Y-%m-%d")
    td = [r for r in runs if time.strftime("%Y-%m-%d", time.localtime(r["t"])) == today]
    ok = [r for r in runs if r["verdict"] == "SUCCESS"]
    secs = [r["seconds"] for r in ok if r["seconds"]]
    return {"runs": len(runs), "success": len(ok), "today": len(td), "today_success": sum(r["verdict"] == "SUCCESS" for r in td),
            "avg_s": round(sum(secs) / len(secs), 1) if secs else None, "best_s": min(secs) if secs else None,
            "recent": runs[-10:][::-1]}


# ----------------------------------------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def js(self, obj):
        self.send(json.dumps(obj).encode(), "application/json")

    def do_POST(self):
        if self.path.startswith("/api/chat"):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            if self.path == "/api/chat":
                text = str(body.get("text", "")).strip()[:2000]
                err = chat_send(text) if text else "empty message"
                return self.js({"ok": err is None, "error": err})
            if self.path == "/api/chat/stop":
                chat_stop()
            elif self.path == "/api/chat/new":
                chat_stop()
                CHAT["session"] = None
                chat_add("system", "new conversation")
            elif self.path == "/api/chat/model" and body.get("model") in ("sonnet", "haiku", "opus"):
                CHAT["model"] = body["model"]
                chat_add("system", f"model: {CHAT['model']}")
            return self.js({"ok": True})
        if self.path == "/api/stop":
            return self.js({"ok": chat_stop()})
        if self.path == "/api/scan":
            if not SCAN["busy"]:
                SCAN.update(busy=True, error=None)
                threading.Thread(target=scan_table, daemon=True).start()
            return self.js({"ok": True})
        if self.path == "/api/world/rename":
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            with WORLD.lock:
                o = WORLD.objs.get(str(body.get("id")))
                if o is not None and not o["known"]:
                    o["name"], o["by_user"] = str(body.get("name", "")).strip().lower()[:40] or None, True
            return self.js({"ok": o is not None})
        if self.path == "/clear" and LIVE["mode"] != "sim":
            f = vision.TRACE_DIR / "events.jsonl"
            if f.exists():
                f.rename(vision.TRACE_DIR / f"events_{int(time.time())}.jsonl")
        self.send(b"", "text/plain", 204)

    def do_GET(self):
        u = urlparse(self.path)
        p, q = u.path, parse_qs(u.query)
        if p.startswith("/live/"):
            key = p.split("/")[2].split(".")[0]
            with LOCK:
                data = LIVE.get(key)
            return self.send(data or b"", "image/jpeg", 200 if data else 503)
        if p.startswith("/img/"):
            f = vision.TRACE_DIR / "img" / Path(p).name
            return self.send(f.read_bytes(), "image/jpeg") if f.exists() else self.send(b"", "text/plain", 404)
        if p == "/api/robot":
            with LOCK:
                r, err = LIVE["robot"], LIVE["robot_err"]
            fresh = r is not None and time.time() - r["t"] < 3
            return self.js({"mode": LIVE["mode"], "ok": fresh, "error": err, "robot": r})
        if p == "/api/scene":
            with LOCK:
                objs, ot, info, lt = list(LIVE["objects"]), LIVE["objects_t"], LIVE["info"], LIVE["t"]
            return self.js({"mode": LIVE["mode"], "objects": objs, "objects_age": round(time.time() - ot, 1) if ot else None,
                            "active": vision.get_active() if LIVE["mode"] != "sim" else SIM_STATE.get("active"),
                            "mat": [[round(float(x), 1), round(float(y), 1)] for x, y in vision.MAT_CM],
                            "story": LIVE.get("story") or story([]), "world": WORLD.snapshot(), "info": info + ("" if time.time() - lt < 5 or LIVE["mode"] == "sim" else "  (cameras stale)")})
        if p == "/api/events":
            since = float(q.get("since", ["0"])[0] or 0)
            return self.js({"events": read_events(since=since)[-150:]})
        if p == "/api/chat":
            since = int(q.get("since", ["0"])[0] or 0)
            with CHAT_LOCK:
                msgs = [m for m in CHAT["msgs"] if m["i"] > since]
            return self.js({"busy": CHAT["busy"], "model": CHAT["model"], "session": CHAT["session"], "msgs": msgs,
                            "sim": LIVE["mode"] == "sim"})
        if p == "/api/mcp":
            import elrobot_mcp
            cfg = json.loads(mcp_config().read_text())
            return self.js({"tools": elrobot_mcp.TOOLS, "instructions": elrobot_mcp.INSTRUCTIONS, "config": cfg})
        if p == "/api/scan":
            return self.js({**{k: SCAN[k] for k in ("busy", "error", "t", "found")},
                            "auto": {k: AUTO[k] for k in ("on", "calls", "max", "status", "change")}})
        if p == "/api/version":  # the page reloads itself when this changes (page files edited / server restarted)
            v = max([STARTED] + [f.stat().st_mtime for f in WEB.glob("*") if f.is_file()])
            return self.js({"v": v})
        if p == "/api/stats":
            return self.js(stats() if LIVE["mode"] != "sim" else {**stats(), "note": "real runs (simulation runs are not counted)"})
        if p == "/api/meshes":
            return self.js({k: {"url": f"/mesh/{k}.bin", "triangles": len(v) // 3} for k, v in MESHES.items()})
        if p.startswith("/mesh/"):
            name = Path(p).name.removesuffix(".bin")
            return self.send(MESHES[name].tobytes(), "application/octet-stream") if name in MESHES else self.send(b"", "text/plain", 404)
        if p == "/state":  # the old page's endpoint (kept for compatibility)
            evs = read_events(80)
            return self.js({"info": LIVE["info"], "events": evs[::-1]})
        f = (WEB / ("dashboard.html" if p in ("/", "/index.html") else p.lstrip("/"))).resolve()
        if WEB.resolve() in f.parents and f.is_file():
            return self.send(f.read_bytes(), MIME.get(f.suffix, "application/octet-stream"))
        self.send(b"not found", "text/plain", 404)


SIM_STATE = {}
MESHES = {}


def story_loop():
    done = set()
    while True:
        try:
            evs = read_events()
            for e in evs:
                d = e.get("data") or {}
                if d.get("type") == "measured" and e["t"] not in done:
                    done.add(e["t"])
                    WORLD.measured(d["label"], d["table"], d["measured"])
            st = story(evs)
            LIVE["last_event_t"] = max((e["t"] for e in evs[-5:]), default=0.0)
            r = LIVE.get("robot")
            if st["held"] and r is not None and r["gripper"] > 70:  # the jaws are open: nothing is held, whatever the log says
                st["held"] = None
            LIVE["story"] = st
        except Exception as e:  # never let a bad log line stop the page
            LIVE["info"] = f"story error: {e}"
        time.sleep(0.3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="SIMULATION: no robot, scripted run (clearly labelled)")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--no-autoscan", action="store_true", help="don't let Claude look at the table by itself when "
                    "the scene changes (the 'Ask Claude' button still works)")
    ap.add_argument("--unknown", action="store_true", help="also track unknown things on the mat and let Claude name "
                    "them (uses your Claude quota; off by default)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", a.port)) == 0:
            sys.exit(f"already running on port {a.port}")
    MESHES.update(load_meshes())
    UNKNOWN["on"] = a.unknown
    AUTO["on"] = not a.no_autoscan
    threading.Thread(target=auto_scan_loop, daemon=True).start()
    threading.Thread(target=story_loop, daemon=True).start()
    threading.Thread(target=namer_loop, daemon=True).start()
    vision.TRACE_DIR.mkdir(parents=True, exist_ok=True)
    if a.sim:
        LIVE["mode"] = "sim"
        for key, f in (("table", "snapshots/look_table.jpg"), ("gripper", "snapshots/look_gripper.jpg")):
            im = cv2.imread(str(vision.ROOT / f))
            if im is not None:
                im = vision.annotate(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), title="SIMULATION: last real frame (no robot connected)")
            else:  # a fresh install has no saved frames
                im = np.full((240, 320, 3), 18, np.uint8)
                cv2.putText(im, f"{key} camera", (92, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(im, "no camera in simulation", (62, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 180, 245), 1, cv2.LINE_AA)
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            LIVE[key] = cv2.imencode(".jpg", im)[1].tobytes()
        LIVE["objects"] = [{"label": "white mug", "x": 15.6, "y": 30.4}, {"label": "black mug", "x": -1.5, "y": 40.9},
                           {"label": "rubiks cube", "x": -30.7, "y": 37.1}, {"label": "orange ball", "x": -12.3, "y": 27.4}]
        LIVE["objects_t"], LIVE["info"] = time.time(), "SIMULATION: no robot connected"

        def sim_runner():
            while True:
                try:
                    sim_loop()
                except Exception as e:  # keep the page alive, show why
                    LIVE["robot_err"] = f"sim: {type(e).__name__}: {e}"
                    time.sleep(2)
        threading.Thread(target=sim_runner, daemon=True).start()

        real = cv2.imread(str(vision.ROOT / "snapshots/look_table.jpg"))
        real = cv2.cvtColor(real, cv2.COLOR_BGR2RGB) if real is not None else None
        LIVE["table_rgb"] = real
        rng = np.random.default_rng(1)

        def sim_objects():
            """Detections like the real table camera: +-1 cm jitter every frame (the world model must hold still),
            no ball while it is held / in the mug, and the unknown things found on the last real frame."""
            while True:
                st = LIVE.get("story") or story([])
                held_or_in = st["held"] or st["release"]
                truth = [{"label": "white mug", "x": 15.6, "y": 30.4}, {"label": "black mug", "x": -1.5, "y": 40.9},
                         {"label": "rubiks cube", "x": -30.7, "y": 37.1}] + ([] if held_or_in else [{"label": "orange ball", "x": -12.3, "y": 27.4}])
                seen = [dict(o, x=o["x"] + rng.normal(0, 0.7), y=o["y"] + rng.normal(0, 0.7)) for o in truth]
                now = time.time()
                WORLD.see_known(seen, now)
                if real is not None and UNKNOWN["on"]:
                    WORLD.see_unknowns(vision.unknown_objects(real), now, real)
                LIVE["objects"], LIVE["objects_t"] = [dict(o, x=round(o["x"], 1), y=round(o["y"], 1)) for o in seen], now
                SIM_STATE["active"] = "orange ball" if st["phase"] in ("pick",) else "white mug" if st["phase"] in ("measure", "carry", "drop", "verify") else None
                time.sleep(0.25)
        threading.Thread(target=sim_objects, daemon=True).start()
    else:
        threading.Thread(target=camera_loop, daemon=True).start()
        threading.Thread(target=state_loop, daemon=True).start()
    try:
        ip = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=3).stdout.split()[0]
    except (OSError, IndexError, subprocess.SubprocessError):
        ip = "localhost"
    print(f"{'SIMULATION ' if a.sim else ''}dashboard: http://localhost:{a.port}  or  http://{ip}:{a.port}   "
          "(Ctrl+C to stop)", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
