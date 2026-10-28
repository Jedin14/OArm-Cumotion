"""Where the tool's tip really is, from the depth camera -- no markers, no
shape model (2026-10-26).

The frame captured for the click was taken with both arms at pre_pick: it is
the scene WITHOUT the arm. With the arm out at the approach point, anything
clearly in front of that background near where the arm model puts the tip is
the tool. Its end is the part furthest out along the tool's direction; the
centre of that end region (corrected for the camera seeing only its near
side) is the tip, sideways. Along the tool it does not matter: the guarded
contact finds the surface.

measure_tip() is pure (numpy only), so it can be checked on recorded frames.
"""

import os
import time
import warnings

import cv2
import numpy as np

from ctm.common import WS, intrinsics

FOREGROUND = 0.015      # m: this much in front of the background is the arm
SEARCH = 0.06           # m: the tool is looked for this close to the predicted tip
END = 0.02              # m: the end region: the last 2 cm along the tool
FLYING = 0.02           # m: a pixel this far off a neighbour is an edge blend
MIN_POINTS = 40         # seen in the search region, else "not visible"
MIN_END = 15            # in the end region
WIDTH = (0.008, 0.08)   # m: a plausible tool width there, else not trusted
DEBUG_DIR = os.path.join(WS, 'native', 'cache', 'tip_vision')


def measure_tip(depth, background, info, t_wo, tip_pred, axis):
    """depth, background: (H, W) metres (0 = no reading), aligned to the
    colour image of `info`; background may be None. t_wo: 4x4 world <- optical.
    tip_pred: where the model puts the tip (world); axis: the tool's
    direction (world, unit). Returns a dict: ok, why, tip (world), centre,
    n, width, end_px (pixels of the end region), tip_px."""
    out = {'ok': False, 'why': '', 'tip': None, 'n': 0, 'width': None,
           'end_px': None, 'tip_px': None}
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    tip_pred = np.asarray(tip_pred, float)
    h, w = depth.shape
    t_ow = np.linalg.inv(t_wo)
    p_cam = t_ow[:3, :3] @ tip_pred + t_ow[:3, 3]
    if p_cam[2] <= 0.1:
        out['why'] = 'the predicted tip is behind or too near the camera'
        return out
    k, dist = intrinsics(info)
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    u0, v0 = fx * p_cam[0] / p_cam[2] + cx, fy * p_cam[1] / p_cam[2] + cy
    out['pred_px'] = (float(u0), float(v0))
    r_px = int(max(fx, fy) * SEARCH / p_cam[2]) + 4
    x0, x1 = max(0, int(u0) - r_px), min(w, int(u0) + r_px + 1)
    y0, y1 = max(0, int(v0) - r_px), min(h, int(v0) + r_px + 1)
    if x0 >= x1 or y0 >= y1:
        out['why'] = 'the predicted tip is outside the picture'
        return out
    d = depth[y0:y1, x0:x1]
    keep = (d > 0.1) & (d < 3.0) & np.isfinite(d)
    # Edge pixels: the D455 blends depth across an edge ("flying pixels"),
    # which put the tool's end centimetres too far (synthetic test,
    # 2026-10-26). Drop any pixel more than 2 cm off a neighbour.
    dz = np.where(keep, d, np.nan)
    pad = np.pad(dz, 1, mode='edge')
    nb = np.stack([pad[1 + dy:pad.shape[0] - 1 + dy, 1 + dx:pad.shape[1] - 1 + dx]
                   for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)      # all-NaN neighbourhoods
        spread = np.nanmax(nb, axis=0) - np.nanmin(nb, axis=0)
    keep &= ~(spread > FLYING)
    if background is not None and background.shape == depth.shape:
        b = background[y0:y1, x0:x1]
        keep &= (b <= 0.1) | ~np.isfinite(b) | (d < b - FOREGROUND)
    vv, uu = np.nonzero(keep)
    if len(uu) < MIN_POINTS:
        out['why'] = f'the tool is not seen near the tip ({len(uu)} points) - blocked by the arm?'
        return out
    uu, vv = uu + x0, vv + y0
    z = depth[vv, uu]
    rays = _rays(uu, vv, k, dist)
    pts = (rays * z[:, None]) @ t_wo[:3, :3].T + t_wo[:3, 3]
    near = np.linalg.norm(pts - tip_pred, axis=1) < SEARCH
    pts, uu, vv = pts[near], uu[near], vv[near]
    out['n'] = int(len(pts))
    if len(pts) < MIN_POINTS:
        out['why'] = f'the tool is not seen near the tip ({len(pts)} points) - blocked by the arm?'
        return out
    s = (pts - tip_pred) @ axis
    s_end = float(np.percentile(s, 98))
    end = s > s_end - END
    if end.sum() < MIN_END:
        out['why'] = f'too little of the tool\'s end is seen ({int(end.sum())} points)'
        return out
    e_pts = pts[end]
    centroid = e_pts.mean(axis=0)
    # sideways frame at the end: e1 away from the camera, e2 across the view
    cam = t_wo[:3, 3]
    view = centroid - cam
    e1 = view - (view @ axis) * axis
    if np.linalg.norm(e1) < 1e-6:
        out['why'] = 'the tool points straight at the camera here'
        return out
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    x2 = (e_pts - centroid) @ e2
    width = float(np.percentile(x2, 98) - np.percentile(x2, 2))
    out['width'] = width
    if not WIDTH[0] <= width <= WIDTH[1]:
        out['why'] = f'what is seen there is {1000 * width:.0f} mm wide - not the tool?'
        return out
    # Across the view (e2): the middle of the outline -- exact for any shape
    # symmetric about the tool's axis, and edge noise only widens it evenly.
    # Towards the camera (e1) only the near side is seen, and edge depths are
    # unreliable (the D455 blends them with the background): take the near
    # surface straight in front of the axis -- the middle strip of the
    # outline, where the camera looks square onto it -- and the axis lies
    # half the tool's width behind it (synthetic test with blended edges,
    # 2026-10-26: the edge depths put it 7-22 mm off).
    x1 = (e_pts - centroid) @ e1
    lo, hi = np.percentile(x2, 2), np.percentile(x2, 98)
    mid2 = (lo + hi) / 2.0
    strip = np.abs(x2 - mid2) <= 0.2 * width
    if strip.sum() < 5:
        out['why'] = 'the middle of the tool is not seen well enough'
        return out
    near1 = np.median(x1[strip])
    centre = centroid + e2 * mid2 + e1 * (near1 + width / 2.0)
    tip = centre - axis * ((centre - tip_pred) @ axis) + axis * s_end
    out.update(ok=True, tip=tip, centre=centre, across=e2, toward=e1,
               end_px=np.c_[uu[end], vv[end]].astype(np.int32))
    tc = t_ow[:3, :3] @ tip + t_ow[:3, 3]
    out['tip_px'] = (float(fx * tc[0] / tc[2] + cx), float(fy * tc[1] / tc[2] + cy))
    return out


# -- the colour image: the tool's outline (2026-10-26) ---------------------------
# On the robot the depth camera sees too little of the tool at 0.3-0.55 m
# (24-43 % of the pixels there, black shadow rings), but the colour image has
# all of it. The outline gives the tool's centre line across the view exactly,
# and its end; the end, as a ray from the camera, is placed where it crosses
# the plane through the predicted tip square to the line in -- an error of the
# model along the line in barely moves it (front: the camera looks nearly
# along the tool; top: about 1:1 with the model's height error).

COLOUR_DIFF = 28        # 0-255: a colour change this big is not noise
SHADOW = (0.35, 0.92)   # darker by this ratio, same hue: a shadow, not the tool
TOOL_WIDTH = (0.012, 0.07)   # m: a plausible width of the tool's end, seen


def _project(p_world, t_ow, k):
    p = t_ow[:3, :3] @ p_world + t_ow[:3, 3]
    return np.array([k[0, 0] * p[0] / p[2] + k[0, 2], k[1, 1] * p[1] / p[2] + k[1, 2]]), p[2]


def measure_tip_rgb(color, background, info, t_wo, tip_pred, axis, way, depth=None,
                    depth_bg=None):
    """color, background: BGR images (background: the same view without the
    arm). way: the direction of the line in. depth, depth_bg: the depth now
    and without the arm (optional, aligned to colour).

    Sideways to the line in there are two directions. ACROSS the view the
    colour outline measures the tip exactly (< 1 mm, synthetic). The OTHER
    one (roughly towards the camera) one picture cannot tell; it comes from
    the depth readings on the tool's body if there are enough of them, else
    it is left as the model has it ('other_from': 'depth' / 'model') -- never
    guessed. Returns a dict like measure_tip(), plus 'across', 'other'
    (unit vectors) and 'across_off', 'other_off' (m, measured - model)."""
    out = {'ok': False, 'why': '', 'tip': None, 'n': 0, 'width': None,
           'end_px': None, 'tip_px': None, 'how': 'colour'}
    if color is None or background is None or color.shape != background.shape:
        out['why'] = 'no colour background for this view'
        return out
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    way = np.asarray(way, float) / np.linalg.norm(way)
    tip_pred = np.asarray(tip_pred, float)
    k, dist = intrinsics(info)
    t_ow = np.linalg.inv(t_wo)
    p0, z0 = _project(tip_pred, t_ow, k)
    if z0 <= 0.1:
        out['why'] = 'the predicted tip is behind or too near the camera'
        return out
    out['pred_px'] = (float(p0[0]), float(p0[1]))
    pb, _zb = _project(tip_pred - axis * 0.10, t_ow, k)
    d_img = p0 - pb
    if np.linalg.norm(d_img) < 3:
        out['why'] = 'the tool points straight at the camera here'
        return out
    d_img /= np.linalg.norm(d_img)
    n_img = np.array([-d_img[1], d_img[0]])
    f = k[0, 0]
    h, w = color.shape[:2]
    R = int(f * 0.09 / z0) + 6
    x0, x1 = max(0, int(p0[0]) - R), min(w, int(p0[0]) + R + 1)
    y0, y1 = max(0, int(p0[1]) - R), min(h, int(p0[1]) + R + 1)
    if x1 - x0 < 10 or y1 - y0 < 10:
        out['why'] = 'the predicted tip is outside the picture'
        return out
    c = color[y0:y1, x0:x1].astype(np.float32) + 1.0
    b = background[y0:y1, x0:x1].astype(np.float32) + 1.0
    # the camera's exposure may have changed: match the brightness first
    gain = float(np.median(c.sum(axis=2) / b.sum(axis=2)))
    c = c / max(gain, 1e-3)
    diff = np.abs(c - b).max(axis=2)
    ratio = c.sum(axis=2) / b.sum(axis=2)
    chroma = np.abs(c / c.sum(axis=2, keepdims=True) - b / b.sum(axis=2, keepdims=True)).max(axis=2)
    shadow = (ratio > SHADOW[0]) & (ratio < SHADOW[1]) & (chroma < 0.04)
    fg = ((diff > COLOUR_DIFF) & ~shadow).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # the part along the tool: within a band about its predicted centre line
    vv, uu = np.nonzero(fg)
    if len(uu) < 30:
        out['why'] = f'the tool is not seen in colour near the tip ({len(uu)} pixels)'
        return out
    px = np.c_[uu + x0, vv + y0].astype(np.float64)
    s = (px - p0) @ d_img                   # along the tool, pixels (+: past the tip)
    t = (px - p0) @ n_img                   # across
    band_px = f * 0.035 / z0                # the model may be some cm off
    body = (np.abs(t) < band_px) & (s > -f * 0.10 / z0) & (s < f * 0.05 / z0)
    if body.sum() < 30:
        out['why'] = f'the tool is not seen in colour near the tip ({int(body.sum())} pixels)'
        return out
    # the centre line across the view: the middle of the outline, row by row
    # along the tool, over the 6 cm behind the end
    s_end = float(np.percentile(s[body], 99))
    seg = body & (s > s_end - f * 0.06 / z0)
    t_mid = (np.percentile(t[seg], 3) + np.percentile(t[seg], 97)) / 2.0
    width_px = float(np.percentile(t[seg], 97) - np.percentile(t[seg], 3))
    width = width_px * z0 / f
    out['n'] = int(seg.sum())
    out['width'] = width
    if not TOOL_WIDTH[0] <= width <= TOOL_WIDTH[1]:
        out['why'] = f'what is seen there is {1000 * width:.0f} mm wide - not the tool?'
        return out
    # the outline's end is the end face's far rim: half the width times how
    # obliquely the camera sees that face, beyond the end's centre
    cam = t_wo[:3, 3]
    view = tip_pred - cam
    view /= np.linalg.norm(view)
    s_tip = s_end - (width_px / 2.0) * abs(float(axis @ view))
    tip_px = p0 + d_img * s_tip + n_img * t_mid
    # the two sideways directions at the tip
    across3 = np.cross(view, way)
    if np.linalg.norm(across3) < 1e-6:
        out['why'] = 'the line in points straight at the camera'
        return out
    across3 /= np.linalg.norm(across3)
    other3 = np.cross(way, across3)
    # ACROSS: the tip pixel as a ray, onto the plane through the predicted tip
    # square to the line in -- its across part does not depend on where along
    # the ray the tip is
    ray = _rays(np.array([tip_px[0]]), np.array([tip_px[1]]), k, dist)[0]
    ray = t_wo[:3, :3] @ ray
    ray /= np.linalg.norm(ray)
    den = float(ray @ way)
    lam = (float((tip_pred - cam) @ ray) if abs(den) < 0.2
           else float((tip_pred - cam) @ way) / den)
    across_off = float((cam + lam * ray - tip_pred) @ across3)
    # OTHER: the depth readings on the tool's body (the outline's pixels, 6 cm
    # behind the end), each moved from the near surface to the axis (half the
    # width further from the camera): their offset from the model's axis
    other_off, other_from, n_depth = 0.0, 'model', 0
    if depth is not None and depth.shape == color.shape[:2]:
        bp = px[seg].astype(np.int64)
        dz = depth[bp[:, 1], bp[:, 0]]
        good = (dz > 0.1) & np.isfinite(dz)
        if depth_bg is not None and depth_bg.shape == depth.shape:
            dbg = depth_bg[bp[:, 1], bp[:, 0]]
            good &= (dbg <= 0.1) | (dz < dbg - 0.01)
        n_depth = int(good.sum())
        if n_depth >= 12:
            rays = _rays(bp[good, 0], bp[good, 1], k, dist) * dz[good][:, None]
            pts = rays @ t_wo[:3, :3].T + t_wo[:3, 3]
            v = pts - cam
            v /= np.linalg.norm(v, axis=1)[:, None]
            pts = pts + v * (width / 2.0)
            rel = pts - tip_pred
            rel = rel - np.outer(rel @ axis, axis)            # off the model's axis
            other_off = float(np.median(rel @ other3))
            other_from = 'depth'
    tip = tip_pred + across3 * across_off + other3 * other_off
    out.update(ok=True, tip=tip, tip_px=(float(tip_px[0]), float(tip_px[1])),
               across=across3, other=other3, across_off=across_off, other_off=other_off,
               other_from=other_from, n_depth=n_depth,
               end_px=px[seg & (s > s_end - f * END / z0)].astype(np.int32))
    return out


def _rays(uu, vv, k, dist):
    """Unit-depth rays (x, y, 1) for many pixels at once, distortion undone."""
    px = np.c_[uu, vv].astype(np.float64).reshape(-1, 1, 2)
    if np.any(np.abs(dist) > 1e-6):
        n = cv2.undistortPoints(px, k, dist).reshape(-1, 2)
    else:
        n = (px.reshape(-1, 2) - [k[0, 2], k[1, 2]]) / [k[0, 0], k[1, 1]]
    return np.c_[n, np.ones(len(n))]


def save_debug(name, **arrays):
    """Keep what a measurement saw (the last 60), for checking it offline."""
    try:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        np.savez_compressed(os.path.join(DEBUG_DIR, f'{time.strftime("%Y%m%d-%H%M%S")}_{name}.npz'),
                            **{k: np.asarray(v, dtype=object if v is None else None)
                               for k, v in arrays.items()})
        files = sorted(os.listdir(DEBUG_DIR))
        for f in files[:-60]:
            os.remove(os.path.join(DEBUG_DIR, f))
    except OSError:
        pass


def replay():
    """python3 -m ctm.tip_vision: measure every saved capture again (after a
    change here) and print what it finds: the tip against the arm model,
    across the view and in the other sideways direction."""
    import types
    files = sorted(f for f in os.listdir(DEBUG_DIR) if f.endswith('.npz')) \
        if os.path.isdir(DEBUG_DIR) else []
    if not files:
        print(f'no captures in {DEBUG_DIR} yet: they are saved by every touch\'s camera check '
              f'and by TIP CHECK')
        return
    for f in files:
        z = np.load(os.path.join(DEBUG_DIR, f), allow_pickle=True)
        if 'color' not in z.files:
            print(f'{f}: depth only (from before the colour check) - skipped')
            continue
        get = lambda key: None if z[key].dtype == object else z[key]
        color = cv2.imdecode(z['color'], cv2.IMREAD_COLOR)
        bg = cv2.imdecode(z['bg_color'], cv2.IMREAD_COLOR)
        info = types.SimpleNamespace(k=list(z['k']), d=list(z['d']))
        r = measure_tip_rgb(color, bg, info, z['t_wo'], z['tip_pred'], z['axis'], z['way'],
                            depth=get('depth'), depth_bg=get('background'))
        if r['ok']:
            print(f'{f}: tip vs model: across {1000 * r["across_off"]:+.1f} mm, other '
                  f'{1000 * r["other_off"]:+.1f} mm ({r["other_from"]}, {r["n_depth"]} depth px); '
                  f'tool seen {1000 * r["width"]:.0f} mm wide, {r["n"]} px')
        else:
            print(f'{f}: {r["why"]}')


if __name__ == '__main__':
    replay()
