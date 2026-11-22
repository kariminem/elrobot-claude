"""ElRobot follower geometry: lowest point of the arm for a given joint reading (virtual floor).

Uses NormaCore's URDF + STL meshes. Joint mapping is the same one Station's 3D view uses:
angle = urdf_lower + position_norm * (urdf_upper - urdf_lower) for motors 1..7. The gripper's opening
(motor 8) barely changes height, so the jaws are placed at mid-travel.

    from arm_geometry import ArmGeometry
    geo = ArmGeometry()
    z, part = geo.lowest_point(position_norm_list_of_8)   # z in metres, in the base frame
    geo.desk_z                                            # bottom of the base = desk surface
"""
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

URDF = Path(__file__).resolve().parent.parent / "vendor/norma-core/hardware/elrobot/simulation/elrobot_follower.urdf"
MOTOR_JOINTS = [f"rev_motor_0{i}" for i in range(1, 8)]
MAX_POINTS_PER_PART = 3000


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(axis, q):
    a = np.asarray(axis, float)
    a /= np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * k + (1 - np.cos(q)) * k @ k


def _tf(xyz=(0, 0, 0), rot=np.eye(3)):
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = xyz
    return t


def _vec(s, default="0 0 0"):
    return np.array([float(v) for v in (s or default).split()])


def _load_stl(path: Path) -> np.ndarray:
    data = path.read_bytes()
    n = struct.unpack_from("<I", data, 80)[0]
    if 84 + n * 50 == len(data):  # binary STL
        tri = np.frombuffer(data, dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]), count=n, offset=84)
        pts = tri["v"].reshape(-1, 3).astype(float)
    else:  # ASCII STL
        pts = np.array([[float(x) for x in line.split()[1:4]] for line in data.decode().splitlines()
                        if line.strip().startswith("vertex")])
    pts = np.unique(np.round(pts, 2), axis=0)
    if len(pts) > MAX_POINTS_PER_PART:  # keep the extremes plus a spread of the rest
        keep = set(np.argsort(pts[:, 2])[:200]) | set(np.linspace(0, len(pts) - 1, MAX_POINTS_PER_PART - 200).astype(int))
        pts = pts[sorted(keep)]
    return pts


class ArmGeometry:
    def __init__(self, urdf: Path = URDF):
        root = ET.parse(urdf).getroot()
        self.joints = {}
        self.children = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            lim = j.find("limit")
            info = dict(
                name=j.get("name"), type=j.get("type"),
                parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                xyz=_vec(o.get("xyz") if o is not None else None), rpy=_vec(o.get("rpy") if o is not None else None),
                axis=_vec(j.find("axis").get("xyz")) if j.find("axis") is not None else None,
                lower=float(lim.get("lower")) if lim is not None else 0.0,
                upper=float(lim.get("upper")) if lim is not None else 0.0)
            self.joints[info["name"]] = info
            self.children.setdefault(info["parent"], []).append(info)
        self.meshes = {}
        for link in root.findall("link"):
            v = link.find("visual")
            if v is None or v.find("geometry/mesh") is None:
                continue
            m = v.find("geometry/mesh")
            o = v.find("origin")
            pts = _load_stl(urdf.parent / m.get("filename")) * _vec(m.get("scale"), "1 1 1")
            t = _tf(_vec(o.get("xyz") if o is not None else None), _rpy(*_vec(o.get("rpy") if o is not None else None)))
            self.meshes[link.get("name")] = (t[:3, :3] @ pts.T).T + t[:3, 3]
        self.desk_z = float(self.meshes["base_link"][:, 2].min())

    def link_transforms(self, position_norm):
        angles = {}
        for name, p in zip(MOTOR_JOINTS, position_norm[:7]):
            j = self.joints[name]
            angles[name] = j["lower"] + float(np.clip(p, 0, 1)) * (j["upper"] - j["lower"])
        tfs = {"base_link": np.eye(4)}
        stack = ["base_link"]
        while stack:
            parent = stack.pop()
            for j in self.children.get(parent, []):
                t = tfs[parent] @ _tf(j["xyz"], _rpy(*j["rpy"]))
                if j["type"] == "revolute":
                    t = t @ _tf(rot=_axis_angle(j["axis"], angles.get(j["name"], 0.0)))
                elif j["type"] == "prismatic":
                    a = j["axis"] / np.linalg.norm(j["axis"])
                    t = t @ _tf(a * (j["lower"] + j["upper"]) / 2)
                tfs[j["child"]] = t
                stack.append(j["child"])
        return tfs

    def lowest_point(self, position_norm, skip=("base_link", "ST3215_1_v1_1", "Joint_01_1", "ST3215_2_v1_1")):
        """Lowest z (m) over the parts that can move down (everything after joint 2), and the part's name."""
        best, part = np.inf, None
        for link, t in self.link_transforms(position_norm).items():
            if link in skip or link not in self.meshes:
                continue
            z = (self.meshes[link] @ t[:3, :3].T + t[:3, 3])[:, 2].min()
            if z < best:
                best, part = z, link
        return float(best), part
