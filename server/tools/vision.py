"""What the robot sees: object detection on the two cameras (plain OpenCV colour/shape rules, no AI), annotated debug
images, and the event trace that tools/vision_dashboard.py shows live (http://<wsl-ip>:8893).

Thresholds measured on the real scene (evening light, dark grey mat):
  table cam: orange ball S 49-96 V 167-218 | white mug S 0-38 V 74-139 | black mug V 1-29 (median 6) | mat V 17-40
             wooden table = dull orange V 115-140 (excluded by the mat-only region and by brightness)
  gripper cam: ball S 42-92 V 210-255 | mat V 57-105 | white paper/mug S < 15 V > 200
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TRACE_DIR = ROOT / "snapshots/vision"
# table-camera pixel (299x224) -> arm x/y in cm: affine fit on 7 spots where the arm really grasped/released (~2.5 cm)
TABLE_PX_TO_CM = np.array([[-0.29401, -0.02313], [-0.05382, 0.26245], [52.00434, -6.54423]])
MAT_ROI = np.array([[66, 108], [246, 108], [299, 224], [0, 224]], np.int32)  # fallback if no saved outline
_mat_file = ROOT / "config/mat_roi.json"
if _mat_file.exists():  # outline detected from the table camera (dark low-saturation region below the back edge)
    MAT_ROI = np.array(json.loads(_mat_file.read_text())["table_px"], np.int32)
MUG_ROI = np.array([[60, 80], [252, 80], [299, 224], [0, 224]], np.int32)    # mug tops stick up above the mat edge
COLORS = {"rubiks cube": (255, 160, 0), "orange ball": (0, 140, 255), "white mug": (255, 255, 255), "black mug": (255, 0, 255),
          "target": (0, 255, 0), "jaws": (0, 255, 255)}


def px_to_cm(u, v):
    x, y = np.array([u, v, 1.0]) @ TABLE_PX_TO_CM
    return float(x), float(y)


def _hsv(rgb):
    h = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return h[..., 0].astype(int), h[..., 1].astype(int), h[..., 2].astype(int)


def _blobs(mask, amin, amax):
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    out = []
    for k in range(1, n):
        a = stats[k, cv2.CC_STAT_AREA]
        if amin <= a <= amax:
            x, y, w, h = (int(stats[k, i]) for i in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
            out.append({"area": int(a), "box": (x, y, w, h), "cx": float(cent[k][0]), "cy": float(cent[k][1])})
    return sorted(out, key=lambda b: -b["area"])


def _near(mask, box, pad=12):
    """How many pixels of `mask` lie in the box grown by `pad` px (e.g. a mug's handle right next to its body)."""
    x, y, w, h = box
    return int(mask[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad].sum())


def _mug_candidates(rgb, body, handle, on_mat):
    """Score blobs of `body` colour: a mug body has a handle of the `handle` colour right next to it. Size helps a
    little, the handle a lot. Only blobs whose bottom stands on the mat (on_mat(x, y) True) count."""
    out = []
    for b in _blobs(body, 80, 5000):
        x, y, w, h = b["box"]
        if w < 10 or h < 10 or not on_mat(x + w / 2, y + h - 1):
            continue
        hn = _near(handle, b["box"]) - _near(handle & body, b["box"], 0)
        b["handle_px"] = int(hn)
        b["score"] = b["area"] + 60 * min(hn, 60)
        out.append(b)
    return sorted(out, key=lambda b: -b["score"])


def _hue_mix(h, s, v, mask):
    """Fractions of the saturated pixels in `mask` per colour family."""
    sel = mask & (s > 90) & (v > 60)
    hh = h[sel]
    n = max(len(hh), 1)
    return {"red": float(((hh <= 3) | (hh >= 165)).sum() / n), "orange": float(((hh > 3) & (hh <= 22)).sum() / n),
            "yellow": float(((hh > 22) & (hh <= 35)).sum() / n), "green": float(((hh > 35) & (hh <= 85)).sum() / n),
            "blue": float(((hh > 85) & (hh <= 130)).sum() / n), "n": int(len(hh))}


def mat_levels(h, s, v):
    """How colourful / bright the bare mat looks in THIS frame (95th percentiles inside the mat). Evening light: S < 40;
    morning daylight with camera gain: S up to ~120. Thresholds for 'clearly not mat' are set above these."""
    roi = np.zeros(h.shape, np.uint8)
    cv2.fillPoly(roi, [MAT_ROI], 1)
    m = cv2.erode(roi, np.ones((9, 9), np.uint8)) > 0
    if m.sum() < 500:
        return 40.0, 60.0
    return float(np.percentile(s[m], 95)), float(np.percentile(v[m], 95))


def find_cube(h, s, v, region, min_px=150, sat_thr=90):
    """Rubik's cube: a compact squarish block of saturated stickers with BLUE/GREEN plus another colour.
    (The ball has no blue/green; the medicine box is mostly white.)"""
    sat = ((s > sat_thr) & (v > 60) & region).astype(np.uint8)
    sat = cv2.morphologyEx(sat, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    best = None
    for b in _blobs(sat > 0, min_px, 20000):
        x, y, w, hh = b["box"]
        if not 0.5 < w / max(hh, 1) < 2.0:
            continue
        box_mask = np.zeros(h.shape, bool)
        box_mask[y:y + hh, x:x + w] = True
        mix = _hue_mix(h, s, v, box_mask & region)
        # a white top face can stick out above the coloured side faces: look for it in the box grown upwards
        ext = np.zeros(h.shape, bool)
        ext[max(0, y - hh // 2):y + hh, max(0, x - w // 5):x + w + w // 5] = True
        white_frac = float(((s < 40) & (v > 115) & ext & region).sum()) / float(w * hh)
        fill = (mix["n"] + white_frac * w * hh) / float(w * hh)
        cool = mix["blue"] + mix["green"]
        warm = mix["red"] + mix["orange"] + mix["yellow"]
        # stickers: blue/green plus a warm colour OR a white face (from above, the top face is often white)
        if cool > 0.2 and (warm > 0.1 or white_frac > 0.12) and fill > 0.4:
            score = b["area"] * fill
            if best is None or score > best["score"]:
                best = {**b, "score": score, "mix": mix}
    return best


def table_objects(rgb):
    """Objects on the mat in the table camera: [{label, box, u, v (table contact point), x, y (cm), area}].
    Mugs must STAND ON THE MAT (bottom edge inside the mat area: the wall/radiator behind the arm never qualifies) and
    are identified by their handle: white mug = white body + RED handle next to it; black mug = black body + WHITE handle."""
    h, s, v = _hsv(rgb)
    roi = np.zeros(rgb.shape[:2], np.uint8)
    cv2.fillPoly(roi, [MAT_ROI], 1)

    def on_mat(px, py):
        return cv2.pointPolygonTest(MAT_ROI.astype(np.float32), (float(px), float(py)), False) >= 0

    found = []
    inside = roi > 0  # NOTHING outside the mat outline is ever considered
    mat_s, mat_v = mat_levels(h, s, v)
    cube = find_cube(h, s, v, inside, sat_thr=max(90.0, mat_s + 15))
    if cube is not None:
        x, y, w, hh = cube["box"]
        found.append({"label": "rubiks cube", **cube, "u": x + w / 2, "v": y + hh - 0.25 * w})
    red = ((h <= 8) | (h >= 165)) & (s > 80) & (v > 40) & inside
    white = (s < 50) & (v > 65) & (v < 240) & inside
    roi_e = cv2.erode(roi, np.ones((9, 9), np.uint8)) > 0  # black = clearly darker than this frame's mat
    mv = v[roi_e] if roi_e.sum() > 500 else np.array([60.0])
    dark_thr = max(14.0, min(0.45 * float(np.percentile(mv, 50)), 0.8 * float(np.percentile(mv, 5))))
    dark = (v < dark_thr) & inside
    for label, body, handle in (("white mug", white, red), ("black mug", dark, white)):
        cands = _mug_candidates(rgb, body, handle, on_mat)
        if label == "white mug":
            # a mug body is SOLID white (a label on a box is fragmented: fill 0.37), big (real mug 450-970 px, the
            # snack box's label 185 px) and its red handle is a small part of it (45/965, 140/893; the snack box's
            # red print: 182 px around a 185 px label)
            cands = [c for c in cands if c["area"] >= 0.55 * c["box"][2] * c["box"][3] and c["area"] >= 300
                     and c["handle_px"] <= 0.5 * c["area"]]
        if cands and (cands[0]["handle_px"] > 3 or cands[0]["area"] > 300):
            b = cands[0]
            x, y, w, hh = b["box"]
            # a cylinder standing on the table: its base centre is a quarter of its width above the lowest pixel
            # a mug seen from here is about as tall as it is wide: a much taller box means the dark mat's front strip
            # merged with the black mug's body below it -> measure from the top of the box
            hh = min(hh, int(1.3 * w))
            b = {**b, "box": (x, y, w, hh)}
            found.append({"label": label, **b, "u": x + w / 2, "v": y + hh - 0.25 * w})
    # the ball LAST: warm colours (peach/orange/red band), roundish, and NOT part of the cube or touching a mug
    # (the cube's orange stickers and the white mug's red handle are warm too)
    excl = np.zeros(h.shape, bool)
    for o in found:
        x, y, w, hh = o["box"]
        g = 14 if "mug" in o["label"] else 4  # a mug's handle (the white mug's is RED) sticks out of its box
        excl[max(0, y - g):y + hh + g, max(0, x - g):x + w + g] = True
    warm = (s > 40) & (v > 110) & ((h <= 22) | (h >= 165)) & inside & ~excl
    warm = cv2.morphologyEx(warm.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    balls = []
    H_, W_ = h.shape
    for b in _blobs(warm, 30, 900):
        x, y, w, hh = b["box"]
        if x <= 2 or x + w >= W_ - 2 or y + hh >= H_ - 2:  # cut off by the image border: a cable / plug, not the ball
            continue
        if not (0.55 < w / max(hh, 1) < 1.8 and b["area"] / float(w * hh) > 0.45):
            continue
        # a ball lies on the DARK mat: the ring around it is mostly dark. Orange print on a white box is not a ball.
        y0, y1, x0, x1 = max(0, y - 5), y + hh + 5, max(0, x - 5), x + w + 5
        ring = np.ones((y1 - y0, x1 - x0), bool)
        ring[5:5 + hh, 5:5 + w] = False
        rv = v[y0:y1, x0:x1][ring[:v[y0:y1, x0:x1].shape[0], :v[y0:y1, x0:x1].shape[1]]]
        if len(rv) and (rv < max(90.0, mat_v + 5)).mean() > 0.5:  # "dark" = no brighter than the bare mat
            balls.append(b)
    if balls:
        b = balls[0]
        found.append({"label": "orange ball", **b, "u": b["cx"], "v": b["cy"]})
    for o in found:
        o["x"], o["y"] = px_to_cm(o["u"], o["v"])
    return found


SURFACE_HEIGHT_CM = {"rubiks cube": 5.7}  # objects the ball can be put ON (top face height)


def ball_on_top(rgb, objs=None):
    """The ball is not on the mat: is it sitting ON an object? Ball colours (warm, one solid patch) in the top part of
    the object's box (the ball merges into the cube's coloured blob, so it is never found as a separate ball).
    Returns (object, warm px) or (None, 0). The gripper camera must still confirm before any grasp."""
    h, s, v = _hsv(rgb)
    warm = ((s > 40) & (v > 110) & ((h <= 22) | (h >= 165))).astype(np.uint8)
    best = (None, 0)
    for o in objs if objs is not None else table_objects(rgb):
        if o["label"] not in SURFACE_HEIGHT_CM:
            continue
        x, y, w, hh = o["box"]
        top = warm[max(0, y - hh // 3):y + int(0.4 * hh), max(0, x - 3):x + w + 3]
        n, _, st, _ = cv2.connectedComponentsWithStats(top)
        big = int(st[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0
        if big > 120 and big > best[1]:
            best = (o, big)
    return best


def _ball_colour(h, s, v):
    """Ball pixels in any light (evening: H 0-4 / 9-18, S 41-99), minus the white mug's deep-red handle (S 95-149)."""
    return (s > 50) & (v > 50) & ((h <= 30) | (h >= 170)) & ~(((h >= 170) | (h <= 2)) & (s > 100))


def ball_off_mat(rgb):
    """After a drop: is the ball lying on the DESK beside the mat (the mat itself is covered by table_objects)?
    Below the desk's far edge only (no wall / radiator / people). Returns the blob or None."""
    h, s, v = _hsv(rgb)
    H, W = h.shape
    roi = np.zeros((H, W), np.uint8)
    cv2.fillPoly(roi, [MAT_ROI], 1)
    m = _ball_colour(h, s, v) & (v > 90) & (roi == 0)
    m[:100, :] = False
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    for b in _blobs(m, 100, 900):  # a ball there is ~250 px; brown wood specks under cables are < 40 px
        x, y, w, hh = b["box"]
        if x <= 2 or x + w >= W - 2 or y + hh >= H - 2:
            continue
        if 0.55 < w / max(hh, 1) < 1.8 and b["area"] / float(w * hh) > 0.45:
            return b
    return None


def ball_vs_circle(rgb, cx, cy, r):
    """Check from the survey pose when the mug's rim can't be detected (a black mug from above hides its handle):
    the survey centred the mug at (cx, cy) with radius r px from this very pose, so the mug is still there.
    'in' = ball colour fills part of that circle or the ball's centre is inside it; 'out' = ball seen clearly outside."""
    h, s, v = _hsv(rgb)
    H, W = h.shape
    inner = np.zeros((H, W), np.uint8)
    cv2.circle(inner, (int(cx), int(cy)), max(int(0.75 * r), 1), 1, -1)
    inner = inner > 0
    warm = (h >= 3) & (h <= 35) & (s >= 25) & (v >= 90)
    frac = float((warm & inner).sum()) / max(int(inner.sum()), 1)
    rim = {"cx": float(cx), "cy": float(cy), "diam": float(2 * r), "box": (int(cx - r), int(cy - r), int(2 * r), int(2 * r)),
           "ball_frac": round(frac, 2)}
    m = _ball_colour(h, s, v)
    m[int(0.80 * H):, :] = False
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    ball = next(({**b, "diam": float(2 * np.sqrt(b["area"] / np.pi))} for b in _blobs(m, 300, 6000)
                 if b["area"] / float(b["box"][2] * b["box"][3]) > 0.5), None)
    if frac > 0.08 or (ball is not None and np.hypot(ball["cx"] - cx, ball["cy"] - cy) < 0.8 * r):
        return "in", rim, ball
    if ball is not None and np.hypot(ball["cx"] - cx, ball["cy"] - cy) > 1.2 * r:
        return "out", rim, ball
    return None, rim, ball


def ball_vs_rim(rgb, kind="white"):
    """Gripper camera looking DOWN at the mug from ~9 cm above the rim after the drop: the whole rim is in view.
    'in' = the ball's centre is inside the rim's ellipse, 'out' = the ball is visible outside it, None = can't tell
    (no rim, or no ball visible: it may be hidden by the mug's wall). Returns (verdict, rim, ball)."""
    H0, W0 = rgb.shape[:2]  # from the survey pose the mug sits near the image centre
    rim = gripper_blob(rgb, kind, near=(W0 / 2, 0.55 * H0) if kind == "dark" else None)
    h, s, v = _hsv(rgb)
    H, W = h.shape
    m = _ball_colour(h, s, v)
    m[int(0.80 * H):, :] = False  # the jaws
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    ball = None
    for b in _blobs(m, 600, 3500):  # the ball from ~9 cm over the rim: round, ~1200 px (handle bits <= 463 px, wood strip fill 0.46)
        x, y, w, hh = b["box"]
        if b["area"] / float(w * hh) > 0.5 and 0.6 < w / max(hh, 1) < 1.6:
            ball = {**b, "diam": float(2 * np.sqrt(b["area"] / np.pi))}
            break
    if rim is not None:
        # inside the mug the ball is lit from above and looks PALE (H 13-21, S 25-63, V 172-233), not orange: look for
        # warm-pale pixels in the inner 72% of the rim ellipse. Ball in: 31% / 46% of that area; empty mug: 0%.
        x, y, w, hh = rim["box"]
        inner = np.zeros((H, W), np.uint8)
        cv2.ellipse(inner, (int(rim["cx"]), int(rim["cy"])), (max(int(0.36 * w), 1), max(int(0.36 * hh), 1)),
                    0, 0, 360, 1, -1)
        inner = inner > 0
        frac = float((((h >= 5) & (h <= 35) & (s >= 25) & (v >= 90)) & inner).sum()) / max(int(inner.sum()), 1)
        rim["ball_frac"] = round(frac, 2)
        if frac > 0.08:
            return "in", rim, ball
    if rim is None or ball is None:
        return None, rim, ball
    x, y, w, hh = rim["box"]
    ex, ey, a, b_ = rim["cx"], rim["cy"], max(w / 2, 1), max(hh / 2, 1)
    inside = ((ball["cx"] - ex) / a) ** 2 + ((ball["cy"] - ey) / b_) ** 2 <= 1.0
    return ("in" if inside else "out"), rim, ball


def unknown_objects(rgb, known=None):
    """Things on the mat that are none of the known objects (a paper, a box, a mouse...): anything clearly brighter,
    more colourful or darker than the mat (mat: V 20-45, S < 40), away from the mat's edge and from the known objects.
    Returns [{x, y (cm, base point), w_cm, d_cm, box, area}]. Only meaningful while the arm is not over the mat."""
    h, s, v = _hsv(rgb)
    H, W = h.shape
    roi = np.zeros((H, W), np.uint8)
    cv2.fillPoly(roi, [MAT_ROI], 1)
    roi = cv2.erode(roi, np.ones((7, 7), np.uint8)) > 0
    mat_s, mat_v = mat_levels(h, s, v)
    fg = roi & ((v > max(80.0, mat_v + 25)) | (s > max(70.0, mat_s + 15)) | (v < 9))
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    for o in (known if known is not None else table_objects(rgb)):
        x, y, w, hh = o["box"]
        g = 16 if "mug" in o.get("label", "") else 6  # a mug's handle sticks out of its box
        fg[max(0, y - g):y + hh + g, max(0, x - g):x + w + g] = False
    out = []
    for b in _blobs(fg, 45, 5000):
        x, y, w, hh = b["box"]
        if w < 6 or hh < 5:
            continue
        u, vb = x + w / 2, y + hh - 0.2 * hh
        cx, cy = px_to_cm(u, vb)
        w_cm = float(np.hypot(*np.subtract(px_to_cm(x, y + hh), px_to_cm(x + w, y + hh))))
        d_cm = float(np.hypot(*np.subtract(px_to_cm(u, y), px_to_cm(u, y + hh))))
        out.append({"x": float(cx), "y": float(cy), "w_cm": round(min(w_cm, 30), 1), "d_cm": round(min(d_cm, 30), 1),
                    "box": (int(x), int(y), int(w), int(hh)), "area": int(b["area"])})
    return out


def gripper_mat_mask(rgb):
    """Region of the gripper image covered by the mat: the convex hull of ALL mat-coloured patches (dark grey,
    unsaturated; a big object in view splits the mat into several). Objects count only if their CENTRE is inside."""
    h, s, v = _hsv(cv2.medianBlur(rgb, 7))  # daylight shows the mat's white speckle: smooth it into one surface
    mat = ((v < 115) & (s < 35)).astype(np.uint8)
    mat[int(0.80 * mat.shape[0]):, :] = 0
    mat = cv2.morphologyEx(mat, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mat)
    keep = [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] > 0.01 * mat.size]
    out = np.zeros(mat.shape, np.uint8)
    if keep and sum(st[k, cv2.CC_STAT_AREA] for k in keep) > 0.08 * mat.size:
        pts = np.vstack([np.column_stack(np.nonzero(lab == k)[::-1]) for k in keep]).astype(np.int32)
        cv2.fillPoly(out, [cv2.convexHull(pts)], 1)
    out[int(0.80 * out.shape[0]):, :] = 1  # the gripper body itself
    return out > 0


def gripper_blob(rgb, kind, big_ok=False, near=None):
    """Best blob of `kind` ('orange' ball, 'white' mug, 'dark' mug) in the gripper camera, only on the mat.
    White mug from above: its inside is shaded (as dark as the mat), only the RIM is a bright ring -> close the ring,
    fit an ellipse, its centre = the opening centre; it must have its RED handle next to it.
    Ball: orange hues, compact, not touching the image edges (wood/background always enter from an edge)."""
    h, s, v = _hsv(rgb)
    H, W = h.shape
    on_mat = gripper_mat_mask(rgb)
    bottom = int(0.80 * H)
    if kind == "white":
        red = ((h <= 2) | (h >= 165)) & (s > 80) & (v > 40)
        ball = cv2.dilate(cv2.morphologyEx(((v > 105) & (s > 20) & (h >= 3) & (h <= 30)).astype(np.uint8),
                                           cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)), np.ones((11, 11), np.uint8)) > 0
        ring = ((s < 40) & (v > 140) & ~ball).astype(np.uint8)  # the ball's white stripes are not a mug rim
        red = red & ~ball
        ring[bottom:, :] = 0
        ring = cv2.morphologyEx(ring, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        best = None
        for c in cv2.findContours(ring, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
            if cv2.contourArea(c) < 300 or len(c) < 20:
                continue
            (cx, cy), axes, _ = cv2.fitEllipse(c)
            x, y, w, hh = cv2.boundingRect(c)
            handle = _near(red, (x, y, w, hh), 15)
            # the handle can face away from the camera: a big clear rim ring is enough (the table camera already told
            # us which mug this is); a visible red handle only adds confidence
            if (handle <= 8 and cv2.contourArea(c) < 700) or max(axes) > 1.6 * W \
                    or not on_mat[min(int(cy), H - 1), min(max(int(cx), 0), W - 1)]:
                continue
            score = cv2.contourArea(c) + 60 * min(handle, 60)
            if best is None or score > best["score"]:
                best = {"cx": float(cx), "cy": float(cy), "diam": float(max(axes)), "area": int(cv2.contourArea(c)),
                        "box": (x, y, w, hh), "handle_px": int(handle), "score": score}
        return best
    if kind == "cube":
        c = find_cube(h, s, v, on_mat & (np.arange(H)[:, None] < bottom), min_px=300)
        if c is None:
            return None
        x, y, w, hh = c["box"]
        # the face pointing UP is what the ball must land on. From above it is the face showing the MOST area: the
        # largest single-colour patch of the cube (the gripper camera sits beside the jaws, so it also sees a side face)
        y0, x0, x1 = max(0, y - hh // 2), max(0, x - w // 5), min(W, x + w + w // 5)
        win = np.zeros(h.shape, bool)
        win[y0:y + hh, x0:x1] = True
        sat = (s > 90) & (v > 60)
        faces = {"white": (s < 40) & (v > 115), "blue": sat & (h > 85) & (h <= 130), "green": sat & (h > 35) & (h <= 85),
                 "yellow": sat & (h > 22) & (h <= 35), "orange": sat & (h > 3) & (h <= 22),
                 "red": sat & ((h <= 3) | (h >= 165))}
        best_face = None
        for name, m in faces.items():
            m = cv2.morphologyEx((m & win).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
            bl = _blobs(m, int(0.08 * w * hh), 4 * w * hh)
            if bl and (best_face is None or bl[0]["area"] > best_face[1]["area"]):
                best_face = (name, bl[0])
        if best_face is not None:
            c["cx"], c["cy"], c["top"] = best_face[1]["cx"], best_face[1]["cy"], f"{best_face[0]} face (largest = top)"
        else:
            c["cx"], c["cy"], c["top"] = x + w / 2, y + w / 2, "box estimate"
        c["diam"] = float(w)
        return c
    if kind == "dark":
        whitish = (s < 40) & (v > 140)
        body = (v < 25).astype(np.uint8)
        body[bottom:, :] = 0
        body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
        cands = [c for c in _mug_candidates(rgb, body, whitish,
                                            lambda px, py: 3 < px < W - 3 and on_mat[min(int(py), H - 1), int(px)])
                 if c["handle_px"] > 8 and c["area"] > 600]
        if not cands:
            return None
        for c in cands:
            x, y, w, hh = c["box"]
            c["cx"], c["cy"], c["diam"] = x + w / 2, y + w / 2, float(w)
        # several dark things can have something white next to them (the jaws, a bag with white print): when we
        # know roughly where the mug should be in the image, take the candidate closest to that spot
        b0 = min(cands, key=lambda c: np.hypot(c["cx"] - near[0], c["cy"] - near[1])) if near else cands[0]
        return b0
    # the ball close under the gripper is in its shadow and looks pale (V 118-176, S 26-93, H 5-20) with white
    # stripes: "brighter than the mat + orange hue", then close the gaps the stripes leave
    mask = (v > 105) & (s > 20) & (h >= 3) & (h <= 30)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
    mask[bottom:, :] = False
    blobs = []
    for bl in _blobs(mask, 120, int(0.6 * mask.size)):
        x, y, w, hh = bl["box"]
        touches_edge = x <= 2 or y <= 2 or x + w >= W - 2
        compact = bl["area"] / float(w * hh) > 0.45
        # big_ok: the ball was just centred and the camera is now very close (ball on a raised object): it fills
        # the view and gets cut by the image edge. A big, fairly compact blob is then still the ball.
        big_close = big_ok and bl["area"] > 0.12 * mask.size and bl["area"] / float(w * hh) > 0.35
        if big_close:  # (it also hides the mat, so the on-mat test can't apply)
            blobs.append(bl)
            continue
        # close under the jaws the ball hides the mat around it: a big, solid, round blob away from the image edges
        # is the ball even if its centre is not on the mat (wood / background always enter from an edge)
        cut = y + hh >= bottom - 2  # at grasp height the jaws' mask cuts the ball's lower part off: it looks flat
        close_ball = (bl["area"] > 0.03 * mask.size and bl["area"] / float(w * hh) > 0.6
                      and 0.6 < w / max(hh, 1) < (2.4 if cut else 1.6)
                      and np.median(s[y:y + hh, x:x + w][mask[y:y + hh, x:x + w]]) > 80)  # the ball S ~135, wood ~53
        colourful = np.median(s[y:y + hh, x:x + w][mask[y:y + hh, x:x + w]]) > 35  # not a faint warm tint (mug inside: 22)
        if not touches_edge and compact and colourful and (on_mat[min(int(bl["cy"]), H - 1), int(bl["cx"])] or close_ball):
            blobs.append(bl)
    if not blobs:
        return None
    b0 = blobs[0]
    b0["diam"] = float(2 * np.sqrt(b0["area"] / np.pi))
    return b0


def annotate(rgb, objs=(), target=None, arrow_from=None, title="", footer="", outline=None):
    """2x upscaled BGR image with boxes, labels, the jaws' target point, the planned correction and (outline=True on
    table images) the mat boundary nothing outside of which is ever used."""
    img = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    if outline:
        cv2.polylines(img, [MAT_ROI * 2], True, (0, 200, 0), 1, cv2.LINE_AA)
    for o in objs:
        x, y, w, h = o["box"]
        col = COLORS.get(o.get("label", ""), (0, 200, 255))
        cv2.rectangle(img, (2 * x, 2 * y), (2 * (x + w), 2 * (y + h)), col, 2)
        txt = o.get("label", "object")
        if "x" in o:
            txt += f"  ({o['x']:+.0f}, {o['y']:.0f}) cm"
        cv2.putText(img, txt, (2 * x, max(14, 2 * y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        if "u" in o:
            cv2.drawMarker(img, (int(2 * o["u"]), int(2 * o["v"])), col, cv2.MARKER_CROSS, 12, 2)
    if target is not None:
        cv2.drawMarker(img, (int(2 * target[0]), int(2 * target[1])), COLORS["target"], cv2.MARKER_TILTED_CROSS, 18, 2)
        cv2.putText(img, "jaws", (int(2 * target[0]) + 10, int(2 * target[1]) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    COLORS["target"], 1, cv2.LINE_AA)
    if arrow_from is not None and target is not None:
        cv2.arrowedLine(img, (int(2 * arrow_from[0]), int(2 * arrow_from[1])), (int(2 * target[0]), int(2 * target[1])),
                        COLORS["target"], 2, tipLength=0.15)
    for i, (txt, y0) in enumerate(((title, 18), (footer, img.shape[0] - 10))):
        if txt:
            cv2.rectangle(img, (0, y0 - 15), (img.shape[1], y0 + 5), (0, 0, 0), -1)
            cv2.putText(img, txt[:90], (6, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def trace(kind, text, image_bgr=None, data=None):
    """Append one step to the trace the dashboard shows (kind: task | detect | think | move | gripper | result | error
    | claude). `data`: optional structured facts (positions in cm etc.) the dashboard's 3D view draws."""
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    (TRACE_DIR / "img").mkdir(exist_ok=True)
    ev = {"t": time.time(), "kind": kind, "text": text}
    if data:
        ev["data"] = data
    if image_bgr is not None:
        name = f"{int(time.time() * 1000)}.jpg"
        cv2.imwrite(str(TRACE_DIR / "img" / name), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        ev["img"] = name
    with open(TRACE_DIR / "events.jsonl", "a") as f:
        f.write(json.dumps(ev) + "\n")


def set_active(label):
    """What the robot is working on right now (the dashboard's gripper view only shows that)."""
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    (TRACE_DIR / "active.json").write_text(json.dumps({"label": label, "t": time.time()}))


def get_active():
    try:
        return json.loads((TRACE_DIR / "active.json").read_text()).get("label")
    except Exception:
        return None


BETWEEN_JAWS = (0.38, 0.62, 0.58, 0.96)  # x0, x1, y0, y1 as fractions of the gripper image


def ball_in_mug_view(rgb):
    """Gripper camera right after releasing over the mug (lifted 3 cm, same angle): a ball that went IN sits on the
    mug's bottom straight below the jaws, i.e. between the jaws in the image; if it bounced out, that window shows the
    empty mug. Counts ball-coloured pixels there. In evening light the ball reads red (H 0-4, S 66-99); pale wood is
    H ~11 but S <= 39, hence S > 50. Returns (pixels, window polygon)."""
    h, s, v = _hsv(rgb)
    H, W = h.shape
    x0, x1, y0, y1 = (int(f * n) for f, n in zip(BETWEEN_JAWS, (W, W, H, H)))
    ballc = (s > 50) & (v > 50) & ((h <= 30) | (h >= 170))
    ballc &= ~(((h >= 170) | (h <= 2)) & (s > 100))  # the white mug's handle: deep red, S 95-149 (ball S <= 99)
    win = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.int32).reshape(-1, 1, 2)
    return int(ballc[y0:y1, x0:x1].sum()), win


def gripper_axis_target_y(dist_cm):
    """Image row where a point `dist_cm` below the fingertips, on the gripper's axis, appears in the gripper camera.
    Fitted on today's grasps (1 cm -> row 163, 6.5 cm -> row 128) with the camera ~10 cm behind the fingertips."""
    return 58.0 + 1155.0 / (max(dist_cm, 0.0) + 10.0)


MAT_CM = np.array([px_to_cm(u, v) for u, v in MAT_ROI], np.float32)  # the mat outline in the arm's x/y (cm)


def over_mat(x_cm, y_cm, margin_cm=-1.5):
    """True if the point (arm frame, cm) is over the mat. The outline comes from the table camera via a mapping
    accurate to ~2.5 cm, so up to 1.5 cm past the estimated edge is allowed (objects lying right at the edge must be
    reachable); anything clearly off the mat (wood, beside the base) stays forbidden."""
    return cv2.pointPolygonTest(MAT_CM.reshape(-1, 1, 2), (float(x_cm), float(y_cm)), True) >= margin_cm
