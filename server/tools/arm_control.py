"""Control the ElRobot follower through NormaCore Station: look through the cameras, read the arm, move it safely.

Used by the MCP server (tools/elrobot_mcp.py, so Claude/Codex can drive the arm) and as a CLI:
    ./tools/arm look                     save both camera views (snapshots/look_*.jpg) + print the arm state
    ./tools/arm state                    joints, gripper-tip position (cm), gripper opening
    ./tools/arm move DX DY DZ [--orient keep|down|free]   move the gripper tip by DX/DY/DZ cm
    ./tools/arm view RIGHT UP DOWN       move in the GRIPPER CAMERA's directions (image right / image up / down), cm
    ./tools/arm to X Y Z [--orient ...]  move the gripper tip to X/Y/Z cm
    ./tools/arm joint NAME DEGREES       rotate one joint (names below) by DEGREES
    ./tools/arm gripper PERCENT          0 = closed, 100 = fully open (stops gently on an object)
    ./tools/arm grasp [--color orange|white|any] [--height CM]   automatic camera-guided grasp from roughly above
    ./tools/arm place X Y                carry the held object to X/Y cm, put it down, release
    ./tools/arm find [--color ...]       what the gripper camera detects (no motion)
    ./tools/arm ready                    fingertips 12 cm above the desk, 22 cm out, pointing down (start pose)
    ./tools/arm home | set-home          go to / save the home (rest) pose (config/home_pose.json)

Coordinates (cm, robot base frame from NormaCore's URDF): origin = centre of the base on the desk,
+z up, +y = straight out from the base when the base joint is centred; x = sideways. The 'tip' is the centre of the fingertips.
Safety on every move: Station health check, follower torque switched on without a jump, speed limit,
no part of the arm below the desk (URDF meshes), stop + hold if a joint is blocked (collision), max distance per call.
The follower must be "(Self-controlled)" in Station (not mirroring the leader).
"""
import argparse
import asyncio
import io
import json
import logging
import math
import re
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
NORMA = ROOT / "vendor/norma-core"
sys.path[:0] = [str(NORMA / "software/station/shared"), str(NORMA)]
from station_py import new_station_client, send_commands  # noqa: E402
from target.gen_python.protobuf.drivers.st3215 import st3215  # noqa: E402
from target.gen_python.protobuf.drivers.usbvideo import usbvideo  # noqa: E402
from target.gen_python.protobuf.station import commands, drivers  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
from arm_geometry import MOTOR_JOINTS, ArmGeometry  # noqa: E402
import vision  # noqa: E402

GOAL_REG, ACCEL_REG, SPEED_REG, TORQUE_REG, PRESENT_REG = 0x2A, 0x29, 0x2E, 0x28, 0x38
JOINT_NAMES = ["base", "shoulder", "elbow", "wrist_flex", "forearm_roll", "wrist_pitch", "wrist_roll", "gripper"]
TICK = 0.1             # s between commands
SPEED = 70             # max ticks per TICK for the furthest joint (~60°/s)
BLOCKED_LAG = 300      # ticks between command and actual position ...
BLOCKED_TIME = 1.0     # ... for this long = something is in the way -> stop and hold
MAX_MOVE_CM = 15.0     # per call
MAX_JOINT_DEG = 45.0   # per call
MUG_Y_CORRECTION_CM = 3.0  # fallback only (gripper survey failed): table camera overshoots a mug (in: -1.5/-2.6 cm; out: -4.4 short, -1.7 long)
HOME_FILE = ROOT / "config/home_pose.json"
log = logging.getLogger("arm")


def signed(d):
    return ((d + 2048) % 4096) - 2048


def station_log() -> Path:
    """NormaCore Station's log (it names the camera queues of the current session). Our project keeps it next to
    the code; anywhere else (e.g. the published plugin) it is found through the running Station process, which
    keeps it open, or ELROBOT_STATION_LOG."""
    if os.environ.get("ELROBOT_STATION_LOG"):
        return Path(os.environ["ELROBOT_STATION_LOG"])
    if (ROOT / "station_data.log").exists():
        return ROOT / "station_data.log"
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            if (proc / "comm").read_text().strip() != "station":
                continue
            for fd in (proc / "fd").iterdir():
                target = os.readlink(fd)
                if target.endswith(".log"):
                    return Path(target)
        except OSError:
            continue
    raise RuntimeError("can't find NormaCore Station's log (is Station running?): set ELROBOT_STATION_LOG to its path")


def load_env():
    """config/hardware.env, each value overridable from the environment as ELROBOT_<NAME> (e.g.
    ELROBOT_FOLLOWER_SERIAL), so the public MCP package works on other people's arms."""
    env = {}
    f = ROOT / "config/hardware.env"
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.split("#")[0].strip()
            if "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    for k in set(env) | {"FOLLOWER_SERIAL", "LEADER_SERIAL"}:
        if os.environ.get("ELROBOT_" + k):
            env[k] = os.environ["ELROBOT_" + k]
    return env


def sync_write(bus, values, address=GOAL_REG, nbytes=2):
    motors = [st3215.ST3215SyncWriteCommand_MotorWrite(motor_id=m, value=int(v).to_bytes(nbytes, "little"))
              for m, v in zip(range(1, 9), values)]
    cmd = st3215.Command(target_bus_serial=bus, sync_write=st3215.ST3215SyncWriteCommand(address=address, motors=motors))
    return commands.DriverCommand(type=drivers.StationCommandType.STC_ST3215_COMMAND, body=cmd.encode())


@dataclass
class State:
    pos: list       # raw present position 0..4095, motors 1..8
    goal: list      # raw goal position register
    ranges: list    # calibrated (min, max) per motor
    torque: list
    errors: list    # error text per motor ("" = fine)

    def norm(self, raw=None):
        raw = self.pos if raw is None else raw
        return [((r - lo) % 4096) / max((hi - lo) % 4096, 1) for r, (lo, hi) in zip(raw, self.ranges)]

    def raw(self, norm):
        return [int(round(lo + min(max(n, 0.0), 1.0) * ((hi - lo) % 4096))) % 4096 for n, (lo, hi) in zip(norm, self.ranges)]


class Arm:
    def __init__(self):
        self.geo = ArmGeometry()
        self.bus = load_env().get("FOLLOWER_SERIAL")  # None: picked in read() (the only arm, or ask which)
        self.client = None
        # Tool point = centre of the FINGERTIPS, fixed in the gripper-base frame (the jaws only slide sideways).
        # Pointing direction = the wrist-roll axis, signed toward the fingers.
        tfs = self.geo.link_transforms([0.5] * 8)
        inv = np.linalg.inv(tfs["Gripper_Base_v1_1"])
        pts = np.vstack([(self.geo.meshes[j] @ (inv @ tfs[j])[:3, :3].T) + (inv @ tfs[j])[:3, 3]
                         for j in ("Gripper_Jaw_01_v1_1", "Gripper_Jaw_02_v1_1")])
        axis = np.array(self.geo.joints["rev_motor_07"]["axis"], dtype=float)
        axis /= np.linalg.norm(axis)
        if (pts.mean(axis=0) @ axis) < 0:
            axis = -axis
        proj = pts @ axis
        self.tip_gb = pts[proj > proj.max() - 0.005].mean(axis=0)
        self.axis_gb = axis
        # gripper camera image axes in the gripper-base frame: image-right = the jaws' closing axis (sign checked
        # on the real arm: with wrist_roll ~0 and pointing down, image-right = +x), image-up = away from the base
        jaw = np.array(self.geo.joints["rev_motor_08_1"]["axis"], dtype=float)
        jaw -= (jaw @ axis) * axis
        jaw /= np.linalg.norm(jaw)
        r0 = tfs["Gripper_Base_v1_1"][:3, :3]
        self.img_right_gb = jaw if (r0 @ jaw)[0] > 0 else -jaw
        up = np.cross(self.img_right_gb, axis)
        self.img_up_gb = up if (r0 @ up)[1] > 0 else -up

    async def connect(self):
        self.client = await new_station_client("localhost", logging.getLogger("station"))
        return self

    # ---------- reading ----------
    async def read(self) -> State:
        qr = self.client.read_from_tail("st3215/inference", offset=b"\x00", limit=1, step=1, buf_size=1)
        entry = await asyncio.wait_for(qr.data.get(), timeout=5)
        s = st3215.InferenceStateReader(memoryview(bytes(entry.Data)))
        buses = s.get_buses() or []
        if self.bus is None:  # no serial configured: fine with exactly one arm on Station, else say which to pick
            serials = [b.get_bus().get_serial_number() for b in buses]
            if len(serials) != 1:
                raise RuntimeError("Station sees " + (f"{len(serials)} arms ({', '.join(serials)})" if serials else "no arm")
                                   + ": set ELROBOT_FOLLOWER_SERIAL to the follower's serial")
            self.bus = serials[0]
        for b in buses:
            if b.get_bus().get_serial_number() != self.bus:
                continue
            motors = sorted(b.get_motors() or [], key=lambda m: m.get_id())
            regs = [bytes(m.get_state() or b"") for m in motors]
            errs = []
            for m in motors:
                e = m.get_error()
                errs.append(f"{e.get_kind().name}: {e.get_description()}" if e is not None and (e.get_description() or e.get_kind() != 0) else "")
            if len(regs) != 8 or any(len(r) <= PRESENT_REG + 1 for r in regs):
                raise RuntimeError("Station isn't reading the follower's 8 motors (Motor ID mismatch?). Run ./restart_station")
            u16 = lambda r, a: r[a] | (r[a + 1] << 8)
            return State(pos=[u16(r, PRESENT_REG) for r in regs], goal=[u16(r, GOAL_REG) for r in regs],
                         ranges=[(int(m.get_range_min()), int(m.get_range_max())) for m in motors],
                         torque=[r[TORQUE_REG] for r in regs], errors=errs)
        raise RuntimeError(f"follower {self.bus} not found in Station (is it plugged in and powered?)")

    def fk(self, norm8):
        """Fingertip centre (m, base frame) and the unit vector the gripper points along."""
        t = self.geo.link_transforms(norm8)["Gripper_Base_v1_1"]
        return t[:3, :3] @ self.tip_gb + t[:3, 3], t[:3, :3] @ self.axis_gb

    def axis_point(self, norm8, plane_z_cm):
        """Where the gripper's axis meets the horizontal plane `plane_z_cm` above the desk: ((x, y) cm, distance along
        the axis in cm). The gripper camera's servo centres objects ON this axis, and a tilted gripper's axis lands
        cm away from the fingertips' own x/y (22° tilt, 6 cm down -> 2.4 cm)."""
        p, d = self.fk(norm8)
        t = ((p[2] - self.geo.desk_z) * 100 - plane_z_cm) / max(-d[2], 0.2)
        return (p[0] * 100 + t * d[0], p[1] * 100 + t * d[1]), t

    def camera_axes(self, norm8):
        """Horizontal unit vectors (base frame) for 'image right' and 'image up' of the gripper camera."""
        r = self.geo.link_transforms(norm8)["Gripper_Base_v1_1"][:3, :3]
        out = []
        for v in (r @ self.img_right_gb, r @ self.img_up_gb):
            v = np.array([v[0], v[1], 0.0])
            out.append(v / max(np.linalg.norm(v), 1e-6))
        return out

    def angles_deg(self, norm8):
        out = []
        for name, n in zip(MOTOR_JOINTS, norm8[:7]):
            j = self.geo.joints[name]
            out.append(math.degrees(j["lower"] + n * (j["upper"] - j["lower"])))
        return out

    def describe(self, st: State) -> str:
        n = st.norm()
        p, d = self.fk(n)
        down = math.degrees(math.asin(max(-1.0, min(1.0, -d[2]))))
        lowest, part = self.geo.lowest_point(n)
        bad = [f"{JOINT_NAMES[i]}: {e}" for i, e in enumerate(st.errors) if e]
        lines = [
            f"fingertips: x={100 * p[0]:+.1f} y={100 * p[1]:+.1f} z={100 * (p[2] - self.geo.desk_z):+.1f} cm above the desk",
            f"gripper points {'DOWN' if down > 60 else 'forward/down' if down > 20 else 'forward' if down > -20 else 'UP'} "
            f"({down:.0f}° below horizontal), direction ({d[0]:+.2f},{d[1]:+.2f},{d[2]:+.2f})",
            f"gripper opening: {100 * n[7]:.0f}% (0 closed, 100 fully open)",
            "joints (deg): " + ", ".join(f"{nm} {a:+.0f}" for nm, a in zip(JOINT_NAMES, self.angles_deg(n))),
            f"lowest arm part: {100 * (lowest - self.geo.desk_z):.1f} cm above the desk ({part})",
            f"torque: {'on' if all(st.torque) else 'OFF (arm is limp)' if not any(st.torque) else 'partly on'}",
        ]
        lines.append("MOTOR ERRORS: " + "; ".join(bad) if bad else "motors: OK")
        return "\n".join(lines)

    async def look(self):
        """Latest JPEG of each camera as Station records it -> [(label, jpeg bytes)], gripper camera first.
        Roles come from config/camera_roles.json (Station queue -> 'gripper'/'table'); for unknown queues the
        gripper camera is the one whose bottom strip is black (the gripper body is always in its view)."""
        log_text = station_log().read_text(errors="replace")
        queues = []
        for q in re.findall(r"queue '/[0-9a-f]+/(usbvideo/[0-9a-f]+)'", log_text):
            if q not in queues:
                queues.append(q)
        roles_file = ROOT / "config/camera_roles.json"
        roles = json.loads(roles_file.read_text()) if roles_file.exists() else {}
        frames = []
        for q in queues[-2:]:
            qr = self.client.read_from_tail(q, offset=b"\x00", limit=1, step=1, buf_size=1)
            entry = await asyncio.wait_for(qr.data.get(), timeout=5)
            if entry is None:
                continue
            data = usbvideo.RxEnvelopeReader(memoryview(bytes(entry.Data))).get_frames().get_frames_data()
            if data:
                jpeg = bytes(data[-1])
                g = np.asarray(Image.open(io.BytesIO(jpeg)).convert("L"), dtype=np.float32)
                frames.append({"queue": q, "jpeg": jpeg, "dark": float((g[int(0.92 * g.shape[0]):] < 40).mean())})
        if not frames:
            raise RuntimeError("no camera frames from Station (cameras attached?)")
        if len(frames) == 2 and not all(f["queue"] in roles for f in frames):
            a, b = sorted(frames, key=lambda f: -f["dark"])
            if a["dark"] - b["dark"] > 0.3:  # confident: remember it
                roles.update({a["queue"]: "gripper", b["queue"]: "table"})
                roles_file.write_text(json.dumps(roles, indent=1))
            else:
                roles = {a["queue"]: "gripper", b["queue"]: "table"}
        names = {"gripper": "gripper camera (mounted on the gripper, jaws at the bottom of the image)",
                 "table": "table camera (fixed, looking at the whole workspace)"}
        frames.sort(key=lambda f: roles.get(f["queue"]) != "gripper")
        return [(names.get(roles.get(f["queue"]), "camera"), f["jpeg"]) for f in frames]

    # ---------- moving ----------
    async def prepare(self) -> State:
        st = await self.read()
        bad = [f"{JOINT_NAMES[i]}: {e}" for i, e in enumerate(st.errors) if e]
        if bad:
            raise RuntimeError("motor errors, not moving: " + "; ".join(bad) + ". Run ./restart_station")
        if not all(st.torque):
            # switch torque on without a jump: goal = present position first
            await send_commands(self.client, [sync_write(self.bus, st.pos)])
            await asyncio.sleep(0.2)
            await send_commands(self.client, [sync_write(self.bus, [1] * 8, TORQUE_REG, 1)])
            await asyncio.sleep(0.5)
            st = await self.read()
            st.goal = list(st.pos)
        await send_commands(self.client, [sync_write(self.bus, [50] * 8, ACCEL_REG, 1),
                                          sync_write(self.bus, [800] * 8, SPEED_REG, 2)])
        # Plan from where the arm really is: if a joint sagged far below its target (gravity, overload protection),
        # re-sync the target to the actual position, otherwise every next move would add to the error and keep the
        # motor straining. The gripper keeps its target (that is its grip on an object).
        if any(abs(signed(g - p)) > 60 for g, p in zip(st.goal[:7], st.pos[:7])):
            st.goal = list(st.pos[:7]) + [st.goal[7]]
            await send_commands(self.client, [sync_write(self.bus, st.goal)])
            await asyncio.sleep(0.2)
        return st

    async def drive(self, st: State, waypoints, speed=SPEED, motors=range(8)) -> str:
        """Stream the command through raw-tick waypoints (each a list of 8). Starts from the current GOAL (so a
        gripper squeezing an object keeps squeezing). Every tick: speed limit, desk check, blocked check."""
        cmd = list(st.goal)
        lag_since = None
        for i, tgt in enumerate(waypoints):
            last = i == len(waypoints) - 1
            while True:
                rem = [signed(t - c) if k in motors else 0 for k, (t, c) in enumerate(zip(tgt, cmd))]
                worst = max(abs(r) for r in rem)
                if worst == 0 or (not last and worst <= speed):
                    break
                scale = min(1.0, speed / worst)
                nxt = [(c + round(r * scale)) % 4096 for c, r in zip(cmd, rem)]
                z_next = self.geo.lowest_point(st.norm(nxt))[0]
                if z_next < self.geo.desk_z and z_next < self.geo.lowest_point(st.norm(cmd))[0]:
                    await self.hold()
                    return "STOPPED: the next step would put part of the arm below the desk surface. Holding."
                cmd = nxt
                await send_commands(self.client, [sync_write(self.bus, cmd)])
                await asyncio.sleep(TICK)
                st = await self.read()
                bad = [f"{JOINT_NAMES[k]}: {e}" for k, e in enumerate(st.errors) if e]
                if bad:
                    return "STOPPED: motor error " + "; ".join(bad) + ". Run ./restart_station"
                lag = max(abs(signed(c - p)) for c, p in zip(cmd[:7], st.pos[:7]))
                if lag > BLOCKED_LAG:
                    lag_since = lag_since or time.time()
                    if time.time() - lag_since > BLOCKED_TIME:
                        await self.hold()
                        j = max(range(7), key=lambda k: abs(signed(cmd[k] - st.pos[k])))
                        return f"STOPPED: the {JOINT_NAMES[j]} joint is blocked (collision or too heavy). Holding where it is."
                else:
                    lag_since = None
        # let it settle
        prev = None
        for _ in range(8):
            await asyncio.sleep(TICK)
            st = await self.read()
            if prev is not None and max(abs(signed(a - b)) for a, b in zip(st.pos, prev)) <= 3:
                break
            prev = st.pos
        return "done"

    async def hold(self):
        st = await self.read()
        goal = list(st.pos)
        goal[7] = st.goal[7]  # keep the gripper's grip
        await send_commands(self.client, [sync_write(self.bus, goal)])

    FREE = [0, 1, 2, 3, 5]  # IK joints: base, shoulder, elbow, wrist_flex, wrist_pitch. The forearm roll (4) and wrist
    #                        roll (6) stay as they are, so the gripper and its camera never twist during a move.

    def ik(self, q8, p_target, d_target=None, w=0.05):
        """Levenberg-Marquardt over the FREE joints, only improving steps accepted. Returns (q8, position error m)."""
        base = np.array(q8, dtype=float)
        q = base[self.FREE].copy()
        n = len(self.FREE)

        def full(qv):
            out = base.copy()
            out[self.FREE] = qv
            return list(out)

        def resid(qv):
            pp, dd = self.fk(full(qv))
            r = pp - p_target
            return (np.concatenate([r, w * (dd - d_target)]) if d_target is not None else r), np.linalg.norm(pp - p_target)

        r, err = resid(q)
        lam = 1e-3
        for _ in range(100):
            if err < 0.0015 and (d_target is None or np.linalg.norm(r[3:]) < 0.1 * w):
                break
            J = np.zeros((len(r), n))
            for k in range(n):
                dq = np.zeros(n)
                dq[k] = 1e-4
                J[:, k] = (resid(q + dq)[0] - r) / 1e-4
            while lam < 1e3:
                step = -np.linalg.solve(J.T @ J + lam * np.eye(n), J.T @ r)
                qn = np.clip(q + np.clip(step, -0.08, 0.08), 0.02, 0.98)
                rn, en = resid(qn)
                if rn @ rn < r @ r:
                    q, r, err, lam = qn, rn, en, max(lam / 3, 1e-6)
                    break
                lam *= 4
            else:
                break
        return full(q), err

    def plan(self, q0, p0, d0, p1, d1, seg_cm=1.0):
        """IK waypoints along the straight line p0->p1, turning the gripper direction gradually from d0 to d1."""
        dist = np.linalg.norm(p1 - p0) * 100
        turn = 0.0 if d1 is None else math.degrees(math.acos(float(np.clip(d0 @ d1, -1, 1))))
        n = max(1, math.ceil(dist / seg_cm), math.ceil(turn / 5))
        q, out = q0, []
        for k in range(1, n + 1):
            f = k / n
            dk = None
            if d1 is not None:
                v = (1 - f) * d0 + f * d1
                dk = v / np.linalg.norm(v)
            q, err = self.ik(q, p0 + (p1 - p0) * f, dk)
            if err > 0.006:
                return None, f, err
            out.append(q)
        return out, 1.0, 0.0

    async def move_tip(self, target_cm=None, delta_cm=None, orient="keep", max_cm=MAX_MOVE_CM) -> str:
        st = await self.prepare()
        q0 = st.norm(st.goal)
        p0, d0 = self.fk(q0)
        if delta_cm is not None:
            p1 = p0 + np.array(delta_cm) / 100
        else:
            p1 = np.array([target_cm[0], target_cm[1], target_cm[2]]) / 100 + np.array([0, 0, self.geo.desk_z])
        dist = np.linalg.norm(p1 - p0) * 100
        if dist > max_cm:
            return f"REFUSED: that is a {dist:.0f} cm move; max {max_cm:.0f} cm per call. Split it into smaller moves."
        if p1[2] - self.geo.desk_z < 0.5 / 100:
            return "REFUSED: target is at or below the desk surface."
        d_t = {"keep": d0, "down": np.array([0.0, 0.0, -1.0]), "free": None}[orient]
        qs, frac, err = self.plan(q0, p0, d0, p1, d_t)
        if qs is None:  # straight line blocked by joint limits: solve the end pose from a neutral seed, move joint-space
            q_end, err = self.ik([0.5] * 4 + [q0[4], 0.5] + list(q0[6:]), p1, d_t)
            if err <= 0.006 and (d_t is None or self.fk(q_end)[1] @ d_t > 0.97):
                qs = [list(np.array(q0) + (np.array(q_end) - np.array(q0)) * k / 10) for k in range(1, 11)]
        if qs is None:
            return (f"REFUSED: can't reach that with orient='{orient}' (stuck {100 * frac:.0f}% along the path, "
                    f"error {100 * err:.1f} cm: out of reach or joint limits). Try a closer target or orient='free'. Nothing moved.")
        ways = []
        for q in qs:
            raw = st.raw(q)
            raw[7] = st.goal[7]
            ways.append(raw)
        res = await self.drive(st, ways)
        st = await self.read()
        p, _ = self.fk(st.norm())
        miss = np.linalg.norm(p - p1) * 100
        return f"{res}. Gripper tip is {miss:.1f} cm from the target.\n" + self.describe(st)

    async def move_in_view(self, right_cm: float, up_cm: float, down_cm: float, orient="keep",
                           along_axis: bool = False) -> str:
        """Move relative to what the GRIPPER CAMERA shows: right_cm toward the image's right edge, up_cm toward
        the image's top edge (both horizontal), down_cm straight down (negative = up). along_axis: go down along the
        gripper's own (tilted) axis instead, so an object centred on the axis STAYS centred while approaching it."""
        st = await self.read()
        rgt, up = self.camera_axes(st.norm(st.goal))
        dn = np.array([0.0, 0.0, -down_cm])
        if along_axis and down_cm > 0:
            ax = self.fk(st.norm(st.goal))[1]
            if ax[2] < -0.3:
                dn = down_cm * ax / -ax[2]  # same height drop, plus the forward slide that keeps the axis on target
        d = right_cm * rgt + up_cm * up + dn
        return await self.move_tip(delta_cm=list(d), orient=orient)

    async def move_joint(self, name: str, degrees: float) -> str:
        if name not in JOINT_NAMES[:7]:
            return f"unknown joint {name!r}; use one of {JOINT_NAMES[:7]}"
        if abs(degrees) > MAX_JOINT_DEG:
            return f"REFUSED: max {MAX_JOINT_DEG:.0f}° per call"
        st = await self.prepare()
        k = JOINT_NAMES.index(name)
        j = self.geo.joints[MOTOR_JOINTS[k]]
        q = st.norm(st.goal)
        q[k] = min(max(q[k] + math.radians(degrees) / (j["upper"] - j["lower"]), 0.0), 1.0)
        tgt = list(st.goal)
        tgt[k] = st.raw(q)[k]
        res = await self.drive(st, [tgt])
        return res + "\n" + self.describe(await self.read())

    async def set_gripper(self, percent: float) -> str:
        st = await self.prepare()
        q = st.norm(st.goal)
        q[7] = min(max(percent, 0), 100) / 100
        target = st.raw(q)[7]
        cmd = st.goal[7]
        still, prev = 0, st.pos[7]
        while True:
            r = signed(target - cmd)
            if r == 0:
                break
            cmd = (cmd + max(-40, min(40, r))) % 4096
            goal = list(st.goal)
            goal[7] = cmd
            await send_commands(self.client, [sync_write(self.bus, goal)])
            await asyncio.sleep(TICK)
            now = await self.read()
            moved = abs(signed(now.pos[7] - prev))
            prev = now.pos[7]
            still = still + 1 if moved <= 3 else 0
            if still >= 4 and abs(signed(cmd - now.pos[7])) > 60:
                # blocked by an object: squeeze gently (small offset) instead of driving the motor into overload
                goal[7] = (now.pos[7] + (80 if signed(cmd - now.pos[7]) > 0 else -80)) % 4096
                await send_commands(self.client, [sync_write(self.bus, goal)])
                now = await self.read()
                return (f"gripper stopped at {100 * now.norm()[7]:.0f}% open: it is HOLDING something "
                        f"(squeezing gently).\n" + self.describe(now))
        await asyncio.sleep(0.4)
        now = await self.read()
        if abs(signed(target - now.pos[7])) > 60:
            # a soft object let the jaws creep on until the end, but they never reached the target: holding it.
            # Back the target off to a gentle squeeze so the motor doesn't strain into overload.
            goal = list(now.goal)
            goal[7] = (now.pos[7] + (80 if signed(target - now.pos[7]) > 0 else -80)) % 4096
            await send_commands(self.client, [sync_write(self.bus, goal)])
            return (f"gripper stopped at {100 * now.norm()[7]:.0f}% open: it is HOLDING something (soft?), "
                    f"squeezing gently.\n" + self.describe(now))
        return "done\n" + self.describe(now)

    # ---------- automatic grasp / place (gripper-camera visual servoing, no AI in the loop) ----------
    # Every step is written to the vision trace (tools/vision.py) with an annotated image and the reason for the
    # decision; tools/vision_dashboard.py shows it live.
    async def camera_images(self):
        out = {}
        for label, jpeg in await self.look():
            out[label.split()[0]] = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
        return out

    async def gripper_image(self) -> np.ndarray:
        return (await self.camera_images())["gripper"]

    async def table_image(self) -> np.ndarray:
        return (await self.camera_images())["table"]

    @staticmethod
    def find_object(rgb: np.ndarray, color: str = "orange"):
        b = vision.gripper_blob(rgb, {"orange": "orange", "white": "white", "black": "dark"}.get(color, "orange"))
        if b is None:
            return None
        return {**b, "label": f"{color} object"}

    def locate_on_table(self, rgb: np.ndarray, label: str = "orange ball"):
        for o in vision.table_objects(rgb):
            if o["label"] == label:
                return o
        return None

    async def fingertip_z(self):
        st = await self.read()
        return (self.fk(st.norm())[0][2] - self.geo.desk_z) * 100, st

    async def servo_over(self, kind: str, label: str, stop_z: float, target_y_far: float = 128.0,
                         target_y_near: float = 163.0, near_z: float = 4.0, far_z: float = 9.0,
                         descend: bool = True, max_steps: int = 16, size_cm: float = 4.0, base_cm: float = 0.0,
                         tol_px: float = 14, leash_cm: float = 12, lost_back: bool = False):
        """Visual servoing with the gripper camera: centre the `kind` blob on the jaws' target point, descending to
        fingertip height `stop_z` (descend=False: only centre). Returns (ok, message)."""
        t0 = time.time()
        vision.set_active(label)
        st0 = await self.read()
        start_xy = self.fk(st0.norm())[0][:2].copy()
        was_near = False
        lost_n = 0
        for it in range(max_steps):
            img = await self.gripper_image()
            obj = vision.gripper_blob(img, kind, big_ok=base_cm > 0 and was_near,
                                      near=(160.0, target_y_far) if kind == "dark" else None)
            z, st = await self.fingertip_z()
            drift = np.linalg.norm(self.fk(st.norm())[0][:2] - start_xy) * 100
            if obj is None:
                if z > 18 + base_cm:
                    vision.trace("detect", f"gripper camera: still no {label} at {z:.1f} cm: give up looking",
                                 vision.annotate(img, title=f"looking for the {label}: not found"))
                    return False, f"no {label} visible in the gripper camera"
                if lost_back:  # survey: the table camera overshoots, so the mug is usually right UNDER the jaws,
                    #               hidden by them: step back toward the base (it slides up into view), never lift
                    lost_n += 1
                    if lost_n > 2:
                        vision.trace("detect", f"gripper camera: still no {label} after stepping back twice: give up",
                                     vision.annotate(img, title=f"looking for the {label}: not found"))
                        return False, f"no {label} visible in the gripper camera"
                    vision.trace("detect", f"gripper camera: no {label} visible → it is probably hidden under the jaws "
                                 "(the table camera overshoots): step 3 cm back toward the base",
                                 vision.annotate(img, title=f"looking for the {label}: not found"))
                    await self.move_in_view(0, -3.0, 0)
                    continue
                vision.trace("detect", f"gripper camera: no {label} visible at fingertip height {z:.1f} cm → lift 2 cm "
                             "to see more", vision.annotate(img, title=f"looking for the {label}: not found"))
                await self.move_tip(delta_cm=[0, 0, 2])
                continue
            if drift > leash_cm:
                vision.trace("error", f"drifted {drift:.0f} cm while chasing the {label}: probably the wrong object. Abort.",
                             vision.annotate(img, [{**obj, "label": label}], title="ABORT: drifted too far"))
                return False, f"drifted {drift:.0f} cm (wrong object?)"
            closeness = min(max((far_z - z) / (far_z - near_z), 0.0), 1.0)
            tx, ty = 160.0, target_y_far + (target_y_near - target_y_far) * closeness
            ex, ey = obj["cx"] - tx, obj["cy"] - ty
            scale = min(max(obj["diam"] / size_cm, 10.0), 25.0)
            # at grasp height a 20 px offset is ~0.6 cm: the closing jaws centre the ball themselves (a 14 px limit
            # there made the servo dither for ~6 s with a jaw resting against a neighbouring object)
            tol = 20 if (descend and z <= stop_z + 0.8) else tol_px
            centred = abs(ex) < tol and abs(ey) < tol
            self.last_servo = (ex, ey, scale)
            self.last_obj = obj
            if centred and (not descend or z <= stop_z + 0.8):
                vision.trace("think", f"{label} centred between the jaws ({ex:+.0f},{ey:+.0f} px) at fingertip height "
                             f"{z:.1f} cm → in position ({time.time() - t0:.0f} s)",
                             vision.annotate(img, [{**obj, "label": label}], (tx, ty), title=f"{label}: IN POSITION"))
                return True, f"centred at {z:.1f} cm in {it + 1} steps"
            near = abs(ex) < 30 and abs(ey) < 30
            was_near = near
            down = 0.0
            if descend:
                down = (min(3.5, z - stop_z) if centred else min(1.5, z - stop_z) if near else
                        (1.0 if z > stop_z + 4 else 0.0))
                if base_cm > 0:  # on a raised object at long reach: small steps, re-centre after each
                    down = min(down, 1.5)
            r, u = 0.8 * ex / scale, -0.8 * ey / scale
            why = ("centred → go down" if centred else "almost centred → correct and go down a little" if near
                   else "off-centre → correct sideways first")
            vision.trace("move", f"{label} is {ex:+.0f} px right / {ey:+.0f} px down of the jaws' target at fingertip "
                         f"height {z:.1f} cm: {why} (right {r:+.1f}, forward {u:+.1f}, down {down:.1f} cm)",
                         vision.annotate(img, [{**obj, "label": label}], (tx, ty), (obj["cx"], obj["cy"]),
                                         title=f"step {it + 1}: {why}"))
            res = await self.move_in_view(r, u, max(down, 0.0), along_axis=base_cm > 0)
            if res.startswith("REFUSED: can't reach"):  # edge of the reach: let the wrist angle change, keep servoing
                vision.trace("think", "can't keep the gripper's angle at this reach → let the wrist tilt (the camera "
                             "re-centres after the move)")
                res = await self.move_in_view(r, u, max(down, 0.0), orient="free")
            if res.startswith(("REFUSED", "STOPPED")):
                vision.trace("error", "move refused/stopped: " + res.splitlines()[0])
                return False, res.splitlines()[0]
        return False, f"could not centre the {label} in {max_steps} steps"

    async def grasp(self, color: str = "orange", grasp_height_cm: float = 2.5, object_width_cm: float = 4.0,
                    align_wrist: bool = False, base_cm: float = 0.0) -> str:
        """From roughly above the object: centre it, descend, close, lift 6 cm. (align_wrist kept for the API; the
        wrist no longer turns: it moved the mat and never helped.)"""
        t0 = time.time()
        label = {"orange": "orange ball", "white": "white object", "black": "black object"}.get(color, color)
        await self.set_gripper(100)
        vision.trace("gripper", "OPEN the gripper fully before going down, so the jaws pass around the object")
        # base_cm: the object rests on something that high (ball on the cube): every height is measured from there
        ok, msg = await self.servo_over(color if color in ("white",) else "orange", label, base_cm + grasp_height_cm,
                                        near_z=4.0 + base_cm, far_z=9.0 + base_cm, size_cm=object_width_cm,
                                        base_cm=base_cm)
        if not ok:
            return f"GRASP ABORTED: {msg}"
        img = await self.gripper_image()
        vision.trace("gripper", f"CLOSE: the {label} is centred and the fingertips are at grasp height "
                     f"({grasp_height_cm:.1f} cm ≈ half the object's height" + (f", above a {base_cm:.1f} cm high surface" if base_cm else "")
                     + "), so closing now grips it around its middle",
                     vision.annotate(img, title="closing the gripper"))
        g = await self.set_gripper(0)
        holding = "HOLDING" in g
        await self.move_tip(delta_cm=[0, 0, 6])
        st = await self.read()
        self.held_grasp_height = grasp_height_cm if holding else None
        img = await self.gripper_image()
        if holding:
            vision.trace("result", f"GRASPED: the jaws stopped at {100 * st.norm()[7]:.0f}% open (blocked by the "
                         f"object = holding it), squeezing gently; lifted 6 cm ({time.time() - t0:.0f} s)",
                         vision.annotate(img, title="GRASPED"),
                         data={"type": "grasped", "label": label, "tip": [round(float(v), 1) for v in
                               (self.fk(st.norm())[0] * 100 - [0, 0, self.geo.desk_z * 100])]})
            return f"GRASPED (gripper {100 * st.norm()[7]:.0f}% open, holding) and lifted 6 cm ({time.time() - t0:.0f} s).\n" + self.describe(st)
        vision.trace("result", "MISSED: the jaws closed all the way, nothing between them", vision.annotate(img, title="MISSED"))
        return "MISSED: the gripper closed on nothing. Look, re-position above the object and try again.\n" + self.describe(st)

    async def pick(self, color: str = "orange", grasp_height_cm: float = 2.5) -> str:
        """One call: find the object in the TABLE camera, go above it, automatic camera-guided grasp, lift."""
        t0 = time.time()
        label = {"orange": "orange ball"}.get(color, color)
        img = await self.table_image()
        objs = vision.table_objects(img)
        found = next((o for o in objs if o["label"] == label), None)
        vision.trace("detect", "table camera: " + (", ".join(f"{o['label']} at ({o['x']:+.0f}, {o['y']:.0f}) cm" for o in objs)
                     or "nothing found"), vision.annotate(img, objs, title="table camera: what I see", outline=True))
        base, on = 0.0, None
        if found is None and label == "orange ball":
            # not on the mat: maybe ON something (it merges into that object's colours in the table camera)
            on, n_warm = vision.ball_on_top(img, objs)
            if on is not None:
                base = vision.SURFACE_HEIGHT_CM[on["label"]]
                found = {"x": on["x"], "y": on["y"]}
                vision.trace("think", f"no {label} on the mat, but {n_warm} ball-coloured pixels sit on TOP of the "
                             f"{on['label']} → the ball is probably ON it ({base:.1f} cm up). Approach from above that "
                             "height; the gripper camera must confirm before the jaws close",
                             vision.annotate(img, [on], title=f"ball on the {on['label']}?", outline=True))
        if found is None:
            return f"REFUSED: no {label} found on the mat (or on top of anything) in the table camera."
        x, y = found["x"], found["y"]
        hover = 10.0 + base
        vision.trace("think", f"target = {label} at ({x:+.1f}, {y:.1f}) cm{(' on the ' + on['label']) if on else ''}"
                     f" → fly {hover:.0f} cm above the table (2 cm short, so the gripper camera sees it ahead), then let "
                     "the gripper camera guide the grasp", data={"type": "target", "label": label, "xy": [round(x, 1), round(y, 1)],
                                                                 "on": on["label"] if on else None})
        st = await self.read()
        if self.fk(st.norm())[0][1] * 100 < 15:
            await self.ready(14)
        z_now, _ = await self.fingertip_z()
        transit = max(min(max(z_now, 14.0), 16.0), base + 12.0)  # travel HIGH (over mugs / the object it sits on)
        vision.trace("think", f"travel at {transit:.0f} cm (above anything standing on the mat), then descend to "
                     f"{hover:.0f} cm over the {label}")
        res = await self.move_tip(target_cm=[x, y - 2.0, transit], orient="free", max_cm=45)
        if not res.startswith(("REFUSED", "STOPPED")):
            res = await self.move_tip(target_cm=[x, y - 2.0, hover], orient="down", max_cm=20)
            if res.startswith(("REFUSED", "STOPPED")):
                res = await self.move_tip(target_cm=[x, y - 2.0, hover], orient="free", max_cm=20)
        if res.startswith(("REFUSED", "STOPPED")):
            vision.trace("error", "can't get above it: " + res.splitlines()[0])
            return f"PICK ABORTED going above the {label} at ({x:.0f}, {y:.0f}) cm: " + res.splitlines()[0]
        g = await self.grasp(color, grasp_height_cm, base_cm=base)
        return f"table camera: {label} at x={x:.1f} y={y:.1f} cm; total {time.time() - t0:.0f} s\n" + g

    async def survey_mug(self, mug: str = "white mug", mug_height_cm: float = 10.0):
        """Fly above the mug (table-camera estimate) with an EMPTY gripper, centre it precisely with the gripper camera
        (the ball would hide it later), and return the exact fingertip x/y to drop at."""
        self.survey_table_estimate = None
        self.survey_pose = None
        img = await self.table_image()
        objs = vision.table_objects(img)
        m = next((o for o in objs if o["label"] == mug), None)
        vision.trace("detect", "table camera: " + (", ".join(f"{o['label']} at ({o['x']:+.0f}, {o['y']:.0f}) cm" for o in objs)
                     or "nothing found"), vision.annotate(img, objs, title=f"looking for the {mug}", outline=True))
        if m is None:
            return None, f"no {mug} found in the table camera"
        self.survey_table_estimate = (m["x"], m["y"])
        hover = mug_height_cm + 6.0
        vision.trace("think", f"{mug} at about ({m['x']:+.1f}, {m['y']:.1f}) cm (table camera, ±2.5 cm, worse at the "
                     f"edges). The opening is only ~4 cm wider than the ball, so first hover {hover:.0f} cm above the "
                     "table (6 cm over the rim) with the EMPTY gripper and let the gripper camera find the rim's centre")
        st = await self.read()
        if self.fk(st.norm())[0][1] * 100 < 15:
            await self.ready(14)
        res = await self.move_tip(target_cm=[m["x"], m["y"] - 4.0, hover], orient="free", max_cm=45)
        if res.startswith(("REFUSED", "STOPPED")):
            vision.trace("error", f"can't hover over the {mug}: " + res.splitlines()[0])
            return None, f"can't reach above the {mug} ({res.splitlines()[0]}); move it closer to the arm"
        kind = "white" if mug.startswith("white") else "dark"
        z_now, _ = await self.fingertip_z()
        if z_now < hover - 1.0:  # long reach sags 2-3 cm: never hang the jaws just over the rim
            vision.trace("move", f"sagged to {z_now:.1f} cm (wanted {hover:.0f}) → lift {hover - z_now:.1f} cm, keep clear of the rim")
            await self.move_tip(delta_cm=[0, 0, hover - z_now], orient="keep")
        st = await self.read()
        _, dist = self.axis_point(st.norm(), mug_height_cm)
        ty = vision.gripper_axis_target_y(dist)  # where the rim's centre appears when it is ON the gripper's axis
        vision.trace("think", f"the mug's opening is {dist:.1f} cm along the gripper's axis: when the axis goes through "
                     f"the rim's centre, the centre shows up at image row {ty:.0f} (camera model)")
        self.last_servo, self.last_obj, self.survey_view = None, None, None
        ok, msg = await self.servo_over(kind, mug, hover, target_y_far=ty, target_y_near=ty, descend=False,
                                        max_steps=8, size_cm=8.0, tol_px=7, leash_cm=8, lost_back=True)
        st = await self.read()
        p = self.fk(st.norm())[0] * 100
        self.survey_pose = [p[0], p[1], p[2] - self.geo.desk_z * 100]  # whole rim in view from here: re-used to check
        tilt = math.degrees(math.acos(min(1.0, -self.fk(st.norm())[1][2])))
        (ax, ay), _ = self.axis_point(st.norm(), mug_height_cm)
        # the last image's leftover offset (<= 7 px when locked), converted to cm and added: no slack left
        rx = ry = 0.0
        if self.last_servo is not None and (ok or msg.startswith("could not centre")):
            ex, ey, scale = self.last_servo
            rgt, up = self.camera_axes(st.norm())
            rx, ry = (ex / scale * rgt + -ey / scale * up)[:2]
            ok = True
        xy = (ax + rx, ay + ry)
        if ok and self.last_obj is not None:  # where the mug sits in the gripper image from this pose (for the check)
            self.survey_view = (float(self.last_obj["cx"]), float(self.last_obj["cy"]), float(self.last_obj["diam"]) / 2)
        far = np.hypot(xy[0] - m["x"], xy[1] - m["y"])
        if ok and far > 9.0:  # the table camera is never THAT far off mid-mat: the gripper locked on something else
            vision.trace("error", f"gripper camera result ({xy[0]:+.1f}, {xy[1]:.1f}) is {far:.0f} cm from the table "
                         "camera's mug: probably another white thing → don't trust it")
            return None, f"gripper-camera reading {far:.0f} cm from the table camera's (wrong object?)"
        vision.trace("result", f"{mug} {'MEASURED' if ok else 'not locked'} ({msg}): fingertips at ({p[0]:+.1f}, "
                     f"{p[1]:.1f}), gripper tilted {tilt:.0f}° → its axis meets the rim at ({ax:+.1f}, {ay:.1f}), "
                     f"+ last {np.hypot(rx, ry):.1f} cm of image offset = opening centre ({xy[0]:+.1f}, {xy[1]:.1f}) cm "
                     f"(table camera said ({m['x']:+.1f}, {m['y']:.1f}))",
                     vision.annotate(await self.gripper_image(), title=f"{mug} position measured"),
                     data={"type": "measured", "label": mug, "table": [round(m["x"], 1), round(m["y"], 1)],
                           "measured": [round(float(xy[0]), 1), round(float(xy[1]), 1)]} if ok else None)
        return (xy if ok else None), msg

    async def drop_into(self, xy, mug: str = "white mug", mug_height_cm: float = 10.0) -> str:
        """Carry the held object over the recorded mug position, 4 cm above the rim, and let go. Never lower."""
        drop_z = mug_height_cm + 1.5 + (getattr(self, "held_grasp_height", None) or 2.5)  # ball bottom 1.5 cm over the rim
        vision.trace("think", f"carry the ball to the recorded {mug} position, fingertips {drop_z:.1f} cm above the "
                     f"table (ball bottom ~1.5 cm above the rim, jaws above the rim) — never lower, so the arm can't hit the mug")
        hold = getattr(self, "held_grasp_height", None) or 2.5
        transit = mug_height_cm + 4.0 + hold  # ball bottom 4 cm above rim height while crossing the mat
        st = await self.read()
        p = self.fk(st.norm())[0] * 100
        z_now = p[2] - self.geo.desk_z * 100
        vision.trace("think", f"carry at {transit:.1f} cm (ball 4 cm above any mug rim: the black mug is on the way), "
                     f"then lower to {drop_z:.1f} cm over the {mug}")
        if z_now < transit:
            await self.move_tip(delta_cm=[0, 0, min(transit - z_now, 14)], orient="free", max_cm=20)
        res = await self.move_tip(target_cm=[xy[0], xy[1], transit], orient="free", max_cm=45)
        if not res.startswith(("REFUSED", "STOPPED")):
            res = await self.move_tip(target_cm=[xy[0], xy[1], drop_z], orient="down", max_cm=20)
        if res.startswith(("REFUSED", "STOPPED")):
            res = await self.move_tip(target_cm=[xy[0], xy[1], drop_z], orient="free", max_cm=45)
        if res.startswith(("REFUSED", "STOPPED")):
            vision.trace("error", "can't carry it over the mug: " + res.splitlines()[0])
            return "DROP ABORTED: " + res.splitlines()[0]
        for k in range(3):  # the arm sags up there: check where it REALLY is (x, y AND height) before letting go
            st = await self.read()
            p = self.fk(st.norm())[0] * 100
            off = np.array([xy[0] - p[0], xy[1] - p[1]])
            dz = drop_z - (p[2] - self.geo.desk_z * 100)
            if np.linalg.norm(off) <= 0.8 and dz <= 0.8:
                break
            vision.trace("move", f"before releasing: fingertips {np.linalg.norm(off):.1f} cm from the mug centre and "
                         f"{max(dz, 0):.1f} cm too low (sag) → correct ({off[0]:+.1f}, {off[1]:+.1f}, {max(dz, 0) + 0.5:+.1f}) cm")
            await self.move_tip(delta_cm=[off[0], off[1], max(dz, 0) + 0.5 if dz > 0.8 else 0], orient="keep")
        st = await self.read()
        p = self.fk(st.norm())[0] * 100
        img = await self.gripper_image()
        vision.trace("gripper", f"OPEN over the {mug}: fingertips at ({p[0]:+.1f}, {p[1]:.1f}) cm vs mug "
                     f"({xy[0]:+.1f}, {xy[1]:.1f}), {p[2] - self.geo.desk_z * 100:.1f} cm up → the ball drops ~1–2 cm "
                     "into the opening", vision.annotate(img, title="releasing"),
                     data={"type": "release", "label": mug, "tip": [round(float(p[0]), 1), round(float(p[1]), 1),
                           round(float(p[2] - self.geo.desk_z * 100), 1)], "target": [round(float(xy[0]), 1), round(float(xy[1]), 1)]})
        await self.set_gripper(100)
        await asyncio.sleep(1.0)  # let it settle (or bounce out) before judging
        self.held_grasp_height = None
        # 1st witness: the gripper camera from the SURVEY pose (where it measured the mug: whole rim in view, not
        # hidden behind the jaws): is the ball inside the rim?
        await self.move_tip(delta_cm=[0, 0, 3], orient="keep")
        pose = getattr(self, "survey_pose", None)
        if pose is not None:
            res = await self.move_tip(target_cm=list(pose), orient="free", max_cm=20)
        else:
            res = await self.move_tip(delta_cm=[0, 0, 3], orient="free")
        img = await self.gripper_image()
        kind = "white" if mug.startswith("white") else "dark"
        g_verdict, rim, gball = vision.ball_vs_rim(img, kind)
        view = getattr(self, "survey_view", None)
        if view is not None and (kind == "dark" or g_verdict is None):  # the survey showed where the mug is in view
            g_verdict, rim, gball = vision.ball_vs_circle(img, *view)
        shown = [dict(rim, label=mug)] if rim is not None else []
        if gball is not None:
            shown.append(dict(gball, label="orange ball"))
        why = {"in": f"the ball's centre is INSIDE the {mug}'s rim", "out": f"the ball is OUTSIDE the {mug}'s rim",
               None: ("can't see the rim" if rim is None else "rim visible, ball not visible (hidden by the mug's wall, "
                      "or gone)")}[g_verdict]
        vision.trace("detect", f"gripper camera, back at the survey pose, looking at the {mug}: {why}",
                     vision.annotate(img, shown, title=f"after the drop: {g_verdict or 'unclear'}"),
                     data={"type": "check", "label": mug, "verdict": g_verdict})
        vision.set_active(None)
        await self.go_home()
        # 2nd witness: the table camera, mat AND desk around it (a ball it can see is not in the mug)
        timg = await self.table_image()
        objs = vision.table_objects(timg)
        ball = next((o for o in objs if o["label"] == "orange ball"), None)
        off = vision.ball_off_mat(timg) if ball is None else None
        if ball is not None or off is not None or g_verdict == "out":
            where = (f"on the mat at ({ball['x']:+.0f}, {ball['y']:.0f}) cm" if ball is not None else
                     "on the desk beside the mat" if off is not None else f"beside the {mug} (gripper camera)")
            vision.trace("result", f"MISSED: the ball is {where}, not in the {mug}",
                         vision.annotate(timg, objs + ([dict(off, label="orange ball")] if off else []),
                                         title="after the drop: MISSED", outline=True))
            return f"MISSED: the ball is {where}, not in the {mug}."
        if g_verdict == "in":
            vision.trace("result", f"SUCCESS: the gripper camera sees the ball inside the {mug}'s rim, and the table "
                         "camera sees no ball anywhere else", vision.annotate(timg, objs, title="after the drop: IN",
                                                                              outline=True))
            return f"SUCCESS: the ball is in the {mug} (inside its rim, seen from above)."
        vision.trace("result", f"UNSURE: not seen inside the {mug}'s rim and not on the mat/desk ({why})",
                     vision.annotate(timg, objs, title="after the drop: UNSURE", outline=True))
        return f"UNSURE: the ball is not seen inside the {mug} ({why}), nor on the mat or desk. Look before trusting it."

    async def put_ball_on(self, target: str = "rubiks cube", target_height_cm: float = 5.7) -> str:
        """Put the orange ball ON TOP of an object, using BOTH cameras: the table camera gives the rough position,
        the gripper camera (pointing straight down above it) confirms and corrects it before setting the ball down."""
        t0 = time.time()
        vision.trace("task", f"TASK: put the orange ball on the {target}", data={"type": "task_start", "task": f"ball → {target}"})
        timg = await self.table_image()
        objs = vision.table_objects(timg)
        tgt = next((o for o in objs if o["label"] == target), None)
        vision.trace("detect", "table camera: " + (", ".join(f"{o['label']} at ({o['x']:+.0f}, {o['y']:.0f}) cm" for o in objs)
                     or "nothing"), vision.annotate(timg, objs, title=f"target: {target}", outline=True))
        if tgt is None:
            return f"FAILED: no {target} found in the table camera"
        x0, y0 = tgt["x"], tgt["y"]
        vision.trace("think", f"table camera: {target} roughly at ({x0:+.1f}, {y0:.1f}) cm (±2–3 cm). Plan: pick the ball, "
                     "carry it high, then let the GRIPPER camera confirm and correct the position from straight above")
        # --- 1) GRIPPER camera survey with the EMPTY gripper: straight down, 6.5 cm above the top, nothing hiding it ---
        st = await self.read()
        if self.fk(st.norm())[0][1] * 100 < 15:
            await self.ready(14)
        survey_z = target_height_cm + 6.5
        res = await self.move_tip(target_cm=[x0, y0, survey_z], orient="down", max_cm=45)
        if res.startswith(("REFUSED", "STOPPED")):
            res = await self.move_tip(target_cm=[x0, y0, survey_z], orient="free", max_cm=45)
        z, _ = await self.fingertip_z()
        st = await self.read()
        _, dist = self.axis_point(st.norm(), target_height_cm)
        ty = vision.gripper_axis_target_y(dist)
        vision.trace("think", f"gripper camera check from {z:.1f} cm (empty gripper, pointing down): when the gripper's "
                     f"axis goes through the {target}'s top ({dist:.1f} cm along it), the top shows up at image row {ty:.0f}")
        ok, msg = await self.servo_over("cube", target, survey_z, target_y_far=ty, target_y_near=ty, descend=False,
                                        max_steps=4, size_cm=5.7)
        st = await self.read()
        (ax, ay), _ = self.axis_point(st.norm(), target_height_cm)  # tilted gripper: the axis, not the fingertips' x/y
        x1, y1 = (ax, ay) if ok else (x0, y0)
        vision.trace("result", f"table camera said ({x0:+.1f}, {y0:.1f}); gripper camera "
                     f"{'CONFIRMED the top face at' if ok else 'could not lock on, keeping'} ({x1:+.1f}, {y1:.1f}) cm — "
                     f"correction {np.hypot(x1 - x0, y1 - y0):.1f} cm", vision.annotate(await self.gripper_image(),
                                                                                    title=f"{target} position recorded"))
        # --- 2) pick the ball ---
        g = await self.pick("orange", 2.5)
        if "GRASPED" not in g:
            return "FAILED at the pick: " + g.splitlines()[0]
        hold = getattr(self, "held_grasp_height", None) or 2.5
        transit = max(16.5, target_height_cm + 4 + hold)  # clear of the mugs (10 cm) too
        z_now, _ = await self.fingertip_z()
        if z_now < transit:
            await self.move_tip(delta_cm=[0, 0, min(transit - z_now, 14)], orient="free", max_cm=20)
        res = await self.move_tip(target_cm=[x1, y1, transit], orient="free", max_cm=45)
        if res.startswith(("REFUSED", "STOPPED")):
            return "FAILED carrying the ball: " + res.splitlines()[0]
        place_z = target_height_cm + hold + 0.5  # ball bottom 0.5 cm above the top: a ball must not be dropped
        res = await self.move_tip(target_cm=[x1, y1, place_z], orient="down", max_cm=20)
        if res.startswith(("REFUSED", "STOPPED")):
            res = await self.move_tip(target_cm=[x1, y1, place_z], orient="free", max_cm=20)
        if res.startswith(("REFUSED", "STOPPED")):
            return "FAILED lowering onto the target: " + res.splitlines()[0]
        for k in range(2):  # sag check: position and height before letting go
            st = await self.read()
            p = self.fk(st.norm())[0] * 100
            off = np.array([x1 - p[0], y1 - p[1]])
            dz = place_z - (p[2] - self.geo.desk_z * 100)
            if np.linalg.norm(off) <= 0.8 and dz <= 0.6:
                break
            await self.move_tip(delta_cm=[off[0], off[1], max(dz, 0) + 0.3 if dz > 0.6 else 0], orient="keep")
        img = await self.gripper_image()
        vision.trace("gripper", f"OPEN: the ball sits ~0.5 cm above the {target}'s top, so it is set down, not dropped",
                     vision.annotate(img, title="setting the ball down"))
        await self.set_gripper(80)
        await asyncio.sleep(0.6)
        await self.move_tip(delta_cm=[0, 0, 4], orient="free")
        self.held_grasp_height = None
        vision.set_active(None)
        await self.go_home()
        timg = await self.table_image()
        objs = vision.table_objects(timg)
        ball = next((o for o in objs if o["label"] == "orange ball"), None)
        tg = next((o for o in objs if o["label"] == target), None)
        ok = False
        if ball is not None and tg is not None:
            x, y, w, hh = tg["box"]
            ok = x - 5 <= ball["cx"] <= x + w + 5 and y - 25 <= ball["cy"] <= y + hh
        elif ball is None:
            ok = None
        if not ok and tg is not None:
            # a ball sitting ON the target merges with it (its red band is as saturated as the stickers): look for the
            # ball's warm colours in the upper part of the target's box, grown upwards by half its height
            hsv = cv2.cvtColor(timg, cv2.COLOR_RGB2HSV)
            hh_, ss_, vv_ = (hsv[..., i].astype(int) for i in range(3))
            x, y, w, hh = tg["box"]
            y0 = max(0, y - hh // 2)
            warm = (ss_ > 40) & (vv_ > 110) & ((hh_ <= 22) | (hh_ >= 165))
            n_warm = int(warm[y0:y + hh // 2, max(0, x - 3):x + w + 3].sum())
            if n_warm > 40:
                ok, ball = True, None
                vision.trace("detect", f"{n_warm} ball-coloured pixels on top of the {target}: the ball is sitting on it")
        verdict = ("SUCCESS: the ball is on top of the " + target if ok else
                   f"MISSED: the ball is at ({ball['x']:+.0f}, {ball['y']:.0f}) cm, not on the {target}" if ok is False and ball
                   else "UNSURE: ball not visible in the table camera")
        vision.trace("result", f"{verdict} ({time.time() - t0:.0f} s)", vision.annotate(timg, objs, title="after", outline=True))
        return f"{verdict}\nTOTAL {time.time() - t0:.0f} s"

    async def put_ball_in_mug(self, mug: str = "white mug", mug_height_cm: float = 10.0, survey=True) -> str:
        t0 = time.time()
        vision.trace("task", f"TASK: put the orange ball in the {mug}", data={"type": "task_start", "task": f"ball → {mug}"})
        # 1) the table camera finds the mug (by its handle, ±2.5 cm, worse at the edges); the GRIPPER camera, empty and
        #    6 cm over the rim, then measures the opening's centre. Done BEFORE the pick: the held ball would hide it.
        # survey=None (auto): a mug CLOSE to the base (table y < 25 cm) gets the gripper-camera survey: there the gripper
        # points almost straight down (accurate) and the table camera is at its worst (9 cm off at the back edge).
        # Farther out the gripper tilts ~34° (survey ~2 cm off) and the table camera minus 3 cm put the ball in.
        if survey is None:
            img = await self.table_image()
            m0 = next((o for o in vision.table_objects(img) if o["label"] == mug), None)
            survey = m0 is not None and m0["y"] < 25.0
            if m0 is not None:
                vision.trace("think", f"{mug} at ({m0['x']:+.1f}, {m0['y']:.1f}) cm in the table camera → " +
                             ("close to the base, where that camera is least accurate: MEASURE it with the gripper camera"
                              if survey else "mid-mat, where the table camera (minus 3 cm) is reliable"))
        if survey:
            xy, msg = await self.survey_mug(mug, mug_height_cm)
        else:
            xy, msg = None, "survey skipped"
            img = await self.table_image()
            objs = vision.table_objects(img)
            m = next((o for o in objs if o["label"] == mug), None)
            vision.trace("detect", "table camera: " + (", ".join(f"{o['label']} at ({o['x']:+.0f}, {o['y']:.0f}) cm"
                                                                  for o in objs) or "nothing"),
                         vision.annotate(img, objs, title=f"target: {mug}", outline=True))
            self.survey_table_estimate = (m["x"], m["y"]) if m else None
            msg = f"no {mug} found in the table camera" if m is None else msg
        if xy is None:
            m = getattr(self, "survey_table_estimate", None)
            if m is None:
                return f"FAILED: {msg}"
            reach = float(np.hypot(m[0], m[1]))
            if reach > 36.0:  # the fallback (table camera - 3 cm) is not good enough out there: don't drop blind
                vision.trace("error", f"the {mug} is {reach:.0f} cm from the arm's base: too far out for the gripper "
                             "camera to look down at it, and the table camera alone is not precise enough to drop into "
                             "it. Not picking the ball. Move the mug closer (within ~35 cm).")
                await self.go_home()
                return (f"FAILED: the {mug} is {reach:.0f} cm from the arm's base, too far for a reliable drop (the gripper "
                        "camera can't look down at it). Move it within ~35 cm of the base and run again.")
            xy = (m[0], m[1] - MUG_Y_CORRECTION_CM)
            vision.trace("think", f"{mug} drop point ({msg}): the table camera's ({m[0]:+.1f}, {m[1]:.1f}) minus "
                         f"{MUG_Y_CORRECTION_CM:.0f} cm (it overshoots mugs) = "
                         f"({xy[0]:+.1f}, {xy[1]:.1f}) cm")
        g = await self.pick("orange", 2.5)
        if "GRASPED" not in g:
            return "FAILED at the pick: " + g.splitlines()[0]
        d = await self.drop_into(xy, mug, mug_height_cm)
        vision.trace("task", f"done in {time.time() - t0:.0f} s: {d}",
                     data={"type": "task_end", "verdict": d.split(":")[0].split()[0], "seconds": round(time.time() - t0, 1)})
        return f"{d}\nTOTAL {time.time() - t0:.0f} s"

    async def ready(self, height_cm: float = 12.0) -> str:
        """Fingertips `height_cm` above the desk, 22 cm in front of the base, pointing straight down: the starting
        pose for looking at the workspace with the gripper camera and for Cartesian moves."""
        st = await self.prepare()
        q0 = st.norm(st.goal)
        h = min(max(height_cm, 6.0), 25.0)
        while True:  # pointing straight down the fingertips reach ~14 cm at most: take the highest reachable height
            p = np.array([0.0, 0.22, h / 100 + self.geo.desk_z])
            q, err = self.ik([0.5] * 4 + [q0[4], 0.5] + list(q0[6:]), p, np.array([0.0, 0.0, -1.0]))
            if err <= 0.005:
                break
            h -= 1.0
            if h < 6.0:
                return f"REFUSED: ready pose not reachable (error {100 * err:.1f} cm)"
        ways = []
        for k in range(1, 11):  # joint-space path, desk-checked every tick
            raw = st.raw(list(np.array(q0) + (np.array(q) - np.array(q0)) * k / 10))
            raw[7] = st.goal[7]
            ways.append(raw)
        res = await self.drive(st, ways, speed=90)
        return res + "\n" + self.describe(await self.read())

    async def go_home(self) -> str:
        if not HOME_FILE.exists():
            return "no home pose saved yet (put the arm in a rest pose and call set_home)"
        home = json.loads(HOME_FILE.read_text())["position_norm"]
        z, _ = await self.fingertip_z()
        if z < 14.0:  # low next to objects: lift STRAIGHT UP first, or the joint-space swing home sweeps through them
            res = await self.move_tip(delta_cm=[0, 0, min(15.0 - z, 12.0)], orient="free", max_cm=15)
            if res.startswith(("REFUSED", "STOPPED")):
                await self.move_tip(delta_cm=[0, 0, 5.0], orient="free")
        st = await self.prepare()
        tgt = st.raw(home)
        tgt[7] = st.goal[7]
        res = await self.drive(st, [tgt], speed=90)
        return res + "\n" + self.describe(await self.read())

    async def set_home(self) -> str:
        st = await self.read()
        HOME_FILE.write_text(json.dumps({"position_norm": [round(x, 4) for x in st.norm()],
                                         "saved": time.strftime("%Y-%m-%d %H:%M"), "follower": self.bus}, indent=1))
        return "home pose saved:\n" + self.describe(st)


async def run(fn_name, *args, **kw):
    arm = await Arm().connect()
    return await getattr(arm, fn_name)(*args, **kw)


def main():
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["look", "state", "move", "view", "to", "joint", "gripper", "ready", "home", "set-home",
                                    "grasp", "place", "find", "locate", "pick", "pickplace", "mug", "on"])
    ap.add_argument("--target", default="rubiks cube")
    ap.add_argument("--target-height", type=float, default=5.7)
    ap.add_argument("--mug", default="white mug")
    ap.add_argument("--mug-height", type=float, default=10.0)
    ap.add_argument("--color", default="any")
    ap.add_argument("--height", type=float, default=2.5, help="grasp: fingertip height for the grip (cm)")
    ap.add_argument("args", nargs="*")
    ap.add_argument("--orient", default="keep", choices=["keep", "down", "free"])
    a = ap.parse_args()

    if a.cmd == "look":
        async def look():
            arm = await Arm().connect()
            shots = await arm.look()
            return shots, arm.describe(await arm.read())
        shots, desc = asyncio.run(look())
        for label, jpeg in shots:
            out = ROOT / f"snapshots/look_{label.split()[0]}.jpg"
            out.write_bytes(jpeg)
            print(f"{label}: {out}")
        print(desc)
    elif a.cmd == "state":
        async def state():
            arm = await Arm().connect()
            return arm.describe(await arm.read())
        print(asyncio.run(state()))
    elif a.cmd == "move":
        print(asyncio.run(run("move_tip", delta_cm=[float(x) for x in a.args], orient=a.orient)))
    elif a.cmd == "view":
        print(asyncio.run(run("move_in_view", *(float(x) for x in a.args[:3]), orient=a.orient)))
    elif a.cmd == "to":
        print(asyncio.run(run("move_tip", target_cm=[float(x) for x in a.args], orient=a.orient)))
    elif a.cmd == "joint":
        print(asyncio.run(run("move_joint", a.args[0], float(a.args[1]))))
    elif a.cmd == "gripper":
        print(asyncio.run(run("set_gripper", float(a.args[0]))))
    elif a.cmd == "grasp":
        print(asyncio.run(run("grasp", a.color, a.height)))
    elif a.cmd == "place":
        print(asyncio.run(run("place_at", float(a.args[0]), float(a.args[1]))))
    elif a.cmd == "on":
        print(asyncio.run(run("put_ball_on", a.target, a.target_height)))
    elif a.cmd == "mug":
        print(asyncio.run(run("put_ball_in_mug", a.mug, a.mug_height)))
    elif a.cmd == "locate":
        async def loc():
            arm = await Arm().connect()
            return vision.table_objects(await arm.table_image())
        print(asyncio.run(loc()))
    elif a.cmd == "pick":
        print(asyncio.run(run("pick", a.color, a.height)))
    elif a.cmd == "pickplace":
        async def pp():
            arm = await Arm().connect()
            t0 = time.time()
            r1 = await arm.pick(a.color, a.height)
            if "GRASPED" not in r1:
                return r1
            r2 = await arm.place_at(-17.0, 27.0)
            r3 = await arm.go_home()
            return f"{r1}\n{r2.splitlines()[0]}\nhome: {r3.splitlines()[0]}\nTOTAL {time.time() - t0:.0f} s"
        print(asyncio.run(pp()))
    elif a.cmd == "find":
        async def find():
            arm = await Arm().connect()
            return Arm.find_object(await arm.gripper_image(), a.color)
        print(asyncio.run(find()))
    elif a.cmd == "ready":
        print(asyncio.run(run("ready", *(float(x) for x in a.args[:1]))))
    elif a.cmd == "home":
        print(asyncio.run(run("go_home")))
    elif a.cmd == "set-home":
        print(asyncio.run(run("set_home")))


if __name__ == "__main__":
    main()
