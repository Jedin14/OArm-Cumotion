#!/usr/bin/env python3
"""Checks on vlm_detector_node's pixel -> world maths. No camera, no model.

    VLM/run_in_vlm_env.sh test_vlm_geometry.py

These are the functions that fail silently: a wrong sign or a transposed
rotation still produces a plausible-looking 3D point, and you only find out when
the arm reaches for the wrong place. The expected values below are worked out by
hand, not from the code, so they catch that.

The node's methods are called against a stub instead of a real node so nothing
here loads PaliGemma or needs a ROS graph.
"""

import math
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np

from vlm_geometry import (
    Intrinsics,
    render_debug,
    axis_from_mask,
    axis_yaw_world,
    build_detections,
    decode_image,
    deproject,
    object_axis_angle,
    parse_paligemma_coordinates,
    quat_to_rot,
    sample_depth,
)

FAILURES = []


def check(label, got, want, tol=1e-6):
    ok = np.allclose(np.asarray(got, dtype=float), np.asarray(want, dtype=float),
                     atol=tol)
    print(f'{"pass" if ok else "FAIL"}  {label}')
    if not ok:
        print(f'        got  {got}')
        print(f'        want {want}')
        FAILURES.append(label)


def fake_msg(encoding, array, step):
    return SimpleNamespace(encoding=encoding, height=array.shape[0],
                           width=array.shape[1], data=array.tobytes(), step=step)


def test_decode_image():
    depth = np.arange(6, dtype=np.uint16).reshape(2, 3)
    check('decode 16UC1', decode_image(fake_msg('16UC1', depth, 6)), depth)

    # rgb8 must come back as BGR, because everything downstream is cv2.
    rgb = np.zeros((1, 2, 3), np.uint8)
    rgb[0, 0] = (10, 20, 30)
    rgb[0, 1] = (40, 50, 60)
    msg = SimpleNamespace(encoding='rgb8', height=1, width=2,
                          data=rgb.tobytes(), step=6)
    check('decode rgb8 -> bgr8', decode_image(msg),
          [[[30, 20, 10], [60, 50, 40]]])

    bgr = np.array([[[1, 2, 3], [4, 5, 6]]], np.uint8)
    msg = SimpleNamespace(encoding='bgr8', height=1, width=2,
                          data=bgr.tobytes(), step=6)
    check('decode bgr8 unchanged', decode_image(msg), bgr)


def test_parse_coordinates():
    # PaliGemma emits ymin, xmin, ymax, xmax normalised to 1024.
    text = '<loc0512><loc0256><loc0768><loc0512> screwdriver'
    got = parse_paligemma_coordinates(text, 848, 480)
    check('parse box', got[0]['box'], [212, 240, 424, 360])
    check('parse centre', got[0]['center'], [318, 300])
    check('parse count', len(got), 1)


def test_deprojection():
    """fx=fy=400, principal point (424, 240), a pixel 20 px right and 20 up."""
    info = SimpleNamespace(k=[400.0, 0.0, 424.0, 0.0, 400.0, 240.0,
                              0.0, 0.0, 1.0])
    intrinsics = Intrinsics.from_camera_info(info)
    check('CameraInfo unpacks to fx, fy, cx, cy',
          list(intrinsics), [400.0, 400.0, 424.0, 240.0])
    got = deproject(intrinsics, 444, 220, 0.8)
    check('deproject', got, [0.04, -0.04, 0.8])


def test_camera_to_world():
    """Rotation of pi about X: optical +z (forward) becomes world -z (down).

    So a point 0.8 m in front of a camera 1.0 m up sits at world z = 0.2, and
    the optical x/y offsets land on world +x / -y.
    """
    rot = quat_to_rot(1.0, 0.0, 0.0, 0.0)
    check('Rx(pi) columns', rot, [[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    trans = np.array([0.5, 0.0, 1.0])
    point_cam = np.array([0.04, -0.04, 0.8])
    check('camera -> world', rot @ point_cam + trans, [0.54, 0.04, 0.2])


def test_depth_sampling():
    """A dead centre pixel must not sink the whole detection.

    The original prototype read depth at the box centre only, so a single dropout
    there -- common on shiny or thin objects -- reported "depth invalid" for an
    otherwise clean detection.
    """
    depth = np.zeros((480, 848), np.uint16)
    depth[200:300, 380:480] = 800                 # object at 0.8 m
    depth[248:252, 428:432] = 0                   # dropout at the centre
    box = [380, 200, 480, 300]

    z, n_px = sample_depth(depth, box, 0.001, 0.15, 3.0)
    check('median depth survives a centre dropout', z, 0.8)
    print(f'pass  sampled {n_px} valid pixels')

    empty = np.zeros((480, 848), np.uint16)
    z_none, n_none = sample_depth(empty, box, 0.001, 0.15, 3.0)
    check('all-invalid depth returns None', [z_none is None, n_none], [True, 0])


def check_equal(label, got, want):
    ok = got == want
    print(f'{"pass" if ok else "FAIL"}  {label}')
    if not ok:
        print(f'        got  {got!r}')
        print(f'        want {want!r}')
        FAILURES.append(label)


def check_axis(label, got, want, tol_deg=6.0):
    """Grasp axes are undirected, so 170 and -10 degrees are the same answer."""
    ok = got is not None and abs((got - want + 90.0) % 180.0 - 90.0) <= tol_deg
    print(f'{"pass" if ok else "FAIL"}  {label}')
    if not ok:
        print(f'        got  {got}')
        print(f'        want {want} (mod 180)')
        FAILURES.append(label)


def draw_bar(canvas, angle_deg, value, length=110, width=18):
    """Bar whose long axis points along (cos a, sin a) in pixel coords, y down."""
    a = math.radians(angle_deg)
    cy, cx = canvas.shape[0] // 2, canvas.shape[1] // 2
    dx, dy = math.cos(a) * length / 2, math.sin(a) * length / 2
    cv2.line(canvas, (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy)),
             value, width)
    return canvas


def test_axis_from_mask():
    """minAreaRect's angle is the short edge whenever w < h; +90 undoes that.

    OpenCV 5 returns rect angles in [-90, 0), so the corrected value can come
    back 180 degrees from the input -- fine for an undirected grasp axis.
    """
    for truth in (0, 30, 45, 60, 90, 135, 170):
        mask = draw_bar(np.zeros((240, 320), np.uint8), truth, 255)
        angle, corners = axis_from_mask(mask, (0, 0))
        check_axis(f'axis_from_mask at {truth} deg', angle, truth)
        check_equal(f'corners returned at {truth} deg',
                    corners is not None and len(corners) == 4, True)

    angle, corners = axis_from_mask(np.zeros((240, 320), np.uint8), (0, 0))
    check_equal('empty mask -> None', (angle, corners), (None, None))

    speck = np.zeros((240, 320), np.uint8)
    speck[100:103, 100:103] = 255                  # 9 px, below min_area
    angle, _ = axis_from_mask(speck, (0, 0))
    check_equal('sub-threshold blob -> None', angle, None)

    # A round object has no long axis, and reporting one anyway is what put
    # the jaws across a roll of tape at -86 degrees in run 1789014831 -- the
    # same object read as -169, -86, 0 and 0 degrees across four detections.
    # The blob is still returned, so the overlay can draw what it measured.
    disc = np.zeros((240, 320), np.uint8)
    cv2.circle(disc, (160, 120), 40, 255, -1)
    angle, corners = axis_from_mask(disc, (0, 0))
    check_equal('a disc reports no axis at all', angle, None)
    check_equal('but is still a blob, so the overlay can show it',
                corners is not None, True)
    # ... and one that is only slightly out of round still reports none,
    # because a hair of elongation is not a grasp axis.
    oval = np.zeros((240, 320), np.uint8)
    cv2.ellipse(oval, (160, 120), (42, 40), 0, 0, 360, 255, -1)
    angle, _ = axis_from_mask(oval, (0, 0))
    check_equal('nor does a 1.05:1 oval', angle, None)
    # A 2:1 one does.
    bar = np.zeros((240, 320), np.uint8)
    cv2.ellipse(bar, (160, 120), (80, 40), 0, 0, 360, 255, -1)
    angle, _ = axis_from_mask(bar, (0, 0))
    check_equal('a 2:1 ellipse still has one', angle is not None, True)


def test_object_axis_prefers_depth():
    """The real failure this replaced: a vertical object read as horizontal.

    A screwdriver standing vertically in the frame has a bright band across it
    where the handle meets the shaft. Otsu on the colour crop locks onto that
    band and reports an axis ~80 degrees off -- which is exactly what the live
    detector produced (bbox 21x117 px, image_angle_deg -10.3). Depth has no
    such texture, so segmenting on it recovers the true vertical axis.
    """
    colour = np.zeros((240, 320, 3), np.uint8)
    depth_mm = np.full((240, 320), 1200, np.uint16)   # background at 1.20 m
    draw_bar(depth_mm, 90, 700)                       # object at 0.70 m
    cv2.rectangle(colour, (140, 130), (180, 150), (255, 255, 255), -1)  # the trap

    box = [148, 63, 172, 177]
    angle, corners, source = object_axis_angle(colour, depth_mm, box, 0.70, 0.001)
    check_equal('depth segmentation used', source, 'depth')
    check_axis('vertical object reads as vertical', angle, 90.0)
    check_equal('corners produced for the debug image',
                corners is not None and len(corners) == 4, True)

    trapped, _, _ = object_axis_angle(colour, np.zeros_like(depth_mm), box,
                                      0.70, 0.001)
    print(f'      (the old intensity path says {trapped:.1f} deg here)')


def test_object_axis_fallbacks():
    blank_depth = np.zeros((240, 320), np.uint16)

    # No usable depth but clear intensity: the intensity path must take over.
    colour = draw_bar(np.zeros((240, 320, 3), np.uint8), 30, (255, 255, 255))
    angle, _, source = object_axis_angle(colour, blank_depth, [100, 80, 220, 160],
                                         0.70, 0.001)
    check_equal('intensity fallback used', source, 'intensity')
    check_axis('intensity fallback angle', angle, 30.0)

    # Nothing to segment at all: fall back to the box's own aspect ratio.
    empty = np.zeros((240, 320, 3), np.uint8)
    tall, _, source = object_axis_angle(empty, blank_depth, [150, 60, 170, 180],
                                        0.70, 0.001)
    check_equal('bbox fallback used', source, 'bbox')
    check_axis('tall box -> vertical axis', tall, 90.0)

    wide, _, _ = object_axis_angle(empty, blank_depth, [60, 150, 180, 170],
                                   0.70, 0.001)
    check_axis('wide box -> horizontal axis', wide, 0.0)

    angle, _, source = object_axis_angle(empty, blank_depth, [10, 10, 10, 10],
                                         0.70, 0.001)
    check_equal('degenerate box -> no angle', (angle, source), (None, 'empty'))


def test_build_detections():
    """The records both consumers read, from one function.

    build_detections is shared by the node and by vlm_detect.py so that the dot
    drawn on the debug image and the coordinate the orchestrator drives to come
    out of the same code. A box with no usable depth has to survive into the
    overlay while staying out of the detections: seeing a box with no depth is
    the whole diagnosis when an object refuses to be picked.
    """
    intrinsics = Intrinsics(400.0, 400.0, 424.0, 240.0)
    rot = quat_to_rot(1.0, 0.0, 0.0, 0.0)          # Rx(pi)
    trans = np.array([0.5, 0.0, 1.0])

    depth = np.zeros((480, 848), np.uint16)
    depth[200:300, 380:480] = 800                  # near object at 0.8 m
    depth[100:160, 600:700] = 1200                 # far object at 1.2 m
    # Third box deliberately over a depth hole.
    raw = [
        {'box': [380, 200, 480, 300], 'center': (430, 250)},
        {'box': [600, 100, 700, 160], 'center': (650, 130)},
        {'box': [50, 50, 120, 120], 'center': (85, 85)},
    ]

    detections, overlay = build_detections(
        np.zeros((480, 848, 3), np.uint8), depth, raw, intrinsics, rot, trans,
        0.001, 0.15, 3.0, 0.03)

    check_equal('two boxes had depth', len(detections), 2)
    check_equal('all three are on the overlay', len(overlay), 3)
    check_equal('the depthless box is on the overlay with depth None',
                [o[1]['depth_m'] for o in overlay].count(None), 1)
    check_equal('nearest first', [d['depth_m'] for d in detections],
                [0.8, 1.2])

    near = detections[0]
    check_equal('centre pixel is carried through', near['center_px'], [430, 250])
    check('camera xyz', near['point_cam'], [0.012, 0.02, 0.8])
    # Rx(pi) sends optical +y to world -y and +z to world -z, then the mount
    # offset is added.
    check('world xyz', near['point'], [0.512, -0.02, 0.2])
    check_equal('every field the orchestrator reads is present',
                sorted(near.keys()),
                ['axis_source', 'axis_yaw', 'bbox_px', 'center_px', 'depth_m',
                 'depth_px', 'image_angle_deg', 'point', 'point_cam'])

    empty, empty_overlay = build_detections(
        np.zeros((480, 848, 3), np.uint8), depth, [], intrinsics, rot, trans,
        0.001, 0.15, 3.0, 0.03)
    check_equal('no boxes gives no detections', (empty, empty_overlay), ([], []))


def test_oversized_box_rejected():
    """A box covering the frame is the model shrugging, not a detection.

    paligemma-3b-pt-224 always answers. Asked for something absent it returns a
    band across the whole view; the centre of that band became a grasp target
    in the middle of a table, out of reach of both arms, and the failure read
    as a reach problem three steps later.
    """
    intrinsics = Intrinsics(400.0, 400.0, 424.0, 240.0)
    depth = np.full((480, 848), 750, np.uint16)      # good depth everywhere
    color = np.zeros((480, 848, 3), np.uint8)

    # The real shape, as observed: the box was the entire frame. On the debug
    # image that reads as a thin border right around the edge -- easy to miss,
    # because the eye goes to the green axis contour drawn inside it.
    band = [{'box': [0, 0, 848, 480], 'center': (423, 239)}]
    detections, overlay = build_detections(
        color, depth, band, intrinsics, np.eye(3), np.zeros(3),
        0.001, 0.15, 3.0, 0.03)
    check_equal('a whole-table band is not a detection', detections, [])
    check_equal('but it stays on the overlay', len(overlay), 1)
    check_equal('with the reason given',
                'covers' in (overlay[0][1].get('rejected') or ''), True)

    # A plausible object-sized box in the same frame must survive.
    small = [{'box': [400, 220, 450, 262], 'center': (425, 241)}]
    detections, _ = build_detections(
        color, depth, small, intrinsics, np.eye(3), np.zeros(3),
        0.001, 0.15, 3.0, 0.03)
    check_equal('an object-sized box is kept', len(detections), 1)

    # And the threshold is adjustable rather than hard-wired.
    detections, _ = build_detections(
        color, depth, band, intrinsics, np.eye(3), np.zeros(3),
        0.001, 0.15, 3.0, 0.03, max_box_fraction=0.0)
    check_equal('max_box_fraction 0 disables the filter', len(detections), 1)


def test_debug_render():
    """The annotated frame really carries a dot and the coordinates.

    Checked by rendering, not by reading the code: the dot is the thing being
    asked for, and it is easy to draw it off-screen or in the background colour
    and not notice.
    """
    intrinsics = Intrinsics(400.0, 400.0, 424.0, 240.0)
    depth = np.zeros((480, 848), np.uint16)
    depth[200:300, 380:480] = 800
    color = np.full((480, 848, 3), 40, np.uint8)
    raw = [{'box': [380, 200, 480, 300], 'center': (430, 250)}]

    detections, overlay = build_detections(
        color, depth, raw, intrinsics, np.eye(3), np.zeros(3),
        0.001, 0.15, 3.0, 0.03)
    canvas = render_debug(color, depth, overlay, 'detect screwdriver',
                          hud=['test render'], depth_scale=0.001)

    check_equal('the canvas matches the frame size', canvas.shape, color.shape)
    check_equal('and is not just a copy of the input',
                bool(np.any(canvas != color)), True)

    # A red dot at the reported centre: render_debug draws (0, 0, 255) there.
    cx, cy = detections[0]['center_px']
    patch = canvas[cy - 3:cy + 4, cx - 3:cx + 4]
    reddest = patch.reshape(-1, 3)
    is_red = ((reddest[:, 2] > 200) & (reddest[:, 0] < 80)
              & (reddest[:, 1] < 80)).any()
    check_equal('a red dot sits on the reported centre pixel', bool(is_red), True)

    # And the coordinate text is on there. Rendered text is not readable back,
    # so check that the region below the box gained light pixels where the
    # xyz line is drawn.
    x1, _, _, y2 = detections[0]['bbox_px']
    text_band = canvas[y2 + 4:y2 + 70, x1:x1 + 260]
    check_equal('coordinate text is drawn under the box',
                bool((text_band > 200).any()), True)

    out = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       'test_debug_render.png')
    cv2.imwrite(out, canvas)
    print(f'pass  wrote {out} -- open it to see the dot and the coordinates')


def test_axis_yaw():
    """Image angle -> world yaw, through the same Rx(pi) camera.

    Optical x maps to world +x, so a horizontal image axis is yaw 0. Optical y
    maps to world -y, so a vertical image axis is yaw -90 deg. If this were
    taken straight from minAreaRect (as the original prototype did) the second case
    would come out +90.
    """
    info = SimpleNamespace(k=[400.0, 0.0, 424.0, 0.0, 400.0, 240.0,
                              0.0, 0.0, 1.0])
    rot = quat_to_rot(1.0, 0.0, 0.0, 0.0)
    intrinsics = Intrinsics.from_camera_info(info)
    got_h = axis_yaw_world(0.0, intrinsics, rot, 0.8)
    got_v = axis_yaw_world(90.0, intrinsics, rot, 0.8)
    check('horizontal image axis -> yaw 0', got_h, 0.0)
    check('vertical image axis -> yaw -90 deg', got_v, -math.pi / 2)

    # An image axis that maps onto world Z has no projection in the XY plane,
    # so there is no yaw to report and the node must say so rather than guess.
    rot_x_up = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    degenerate = axis_yaw_world(0.0, intrinsics, rot_x_up, 0.8)
    check('axis along world Z -> no yaw', [degenerate is None], [True])


def main():
    test_decode_image()
    test_parse_coordinates()
    test_deprojection()
    test_camera_to_world()
    test_depth_sampling()
    test_axis_from_mask()
    test_object_axis_prefers_depth()
    test_object_axis_fallbacks()
    test_axis_yaw()
    test_build_detections()
    test_oversized_box_rejected()
    test_debug_render()

    print()
    if FAILURES:
        print(f'{len(FAILURES)} failure(s): {", ".join(FAILURES)}')
        return 1
    print('all geometry checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
