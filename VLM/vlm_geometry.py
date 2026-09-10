#!/usr/bin/env python3
"""Detection parsing, depth geometry and the debug overlay. No ROS, no torch.

Split out of vlm_detector_node.py so that the parts with no heavy dependencies
can be imported -- and tested -- without them. The node needs rclpy, torch and
transformers; none of this does, and requiring an 8 GB venv to check a
deprojection was absurd. Importing the node module used to pull in transformers,
which on a system python picks up whatever PIL is installed and dies with
"module 'PIL.Image' has no attribute 'Resampling'" before a single check ran.

Needs only numpy and cv2, both of which the system python has, so:

    python3 VLM/test_vlm_geometry.py

works with nothing sourced and nothing activated.

Everything here is shared by vlm_detector_node.py and vlm_detect.py, which is
the point: the dot drawn on a debug frame and the coordinate the orchestrator
drives to come out of the same functions and cannot drift apart.
"""

import math
import re
from typing import NamedTuple

import cv2
import numpy as np

LOC_PATTERN = re.compile(r'<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>')


def quat_to_rot(x, y, z, w):
    """Quaternion -> 3x3 rotation matrix. Avoids a tf2_geometry_msgs/PyKDL dep."""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def decode_image(msg):
    """sensor_msgs/Image -> numpy array, for the encodings this pipeline emits."""
    if msg.encoding in ('16UC1', 'mono16'):
        return np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width)
    if msg.encoding == 'rgb8':
        arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        return arr[:, :, ::-1].copy()          # to BGR, what cv2 expects
    if msg.encoding == 'bgr8':
        return np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3).copy()
    raise ValueError(f'unsupported encoding {msg.encoding}')


def parse_paligemma_coordinates(output_text, img_width, img_height):
    """Unchanged from the original prototype: <locNNNN> quadruples -> pixel boxes."""
    detections = []
    for match in LOC_PATTERN.findall(output_text):
        ymin, xmin, ymax, xmax = [int(v) / 1024.0 for v in match]
        x1, y1 = int(xmin * img_width), int(ymin * img_height)
        x2, y2 = int(xmax * img_width), int(ymax * img_height)
        detections.append({
            'box': [x1, y1, x2, y2],
            'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
        })
    return detections


def clamp_box(box, shape):
    x1, y1, x2, y2 = box
    h, w = shape[:2]
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


# How much longer the long side has to be before the angle means anything.
#
# A round object's minAreaRect is square to within noise, and its angle is
# then whatever the contour happened to do that frame. Measured on one roll
# of tape, run 1789014831, four detections of the same object on the same
# table: world yaw -169, -86, 0 and 0 degrees. The attempt that got -86 put
# the jaws across a diameter they cannot span and closed on nothing.
#
# 1.15 is deliberately low. It is not trying to judge shape -- anything
# genuinely elongated clears it easily (a screwdriver crop is 4:1 or more)
# and anything near square has no long axis to report.
MIN_AXIS_ASPECT = 1.15


def axis_from_mask(mask, origin, min_area=30, min_aspect=MIN_AXIS_ASPECT):
    """Long axis of the biggest blob in `mask`, in degrees, plus its corners.

    The angle is such that the axis direction is (cos a, sin a) in pixel
    coordinates. minAreaRect's own angle describes the rect's first edge, which
    is the short one whenever width < height, so the +90 puts it back on the
    long axis. Verified against OpenCV 5, whose rect angles run in [-90, 0):
    every orientation comes back correct modulo 180 degrees, which is all an
    undirected grasp axis needs.

    Three outcomes, and the caller needs to tell them apart:

      (angle, corners)  an elongated blob, and which way it lies
      (None, corners)   a blob too square for its angle to mean anything
      (None, None)      no blob worth measuring
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None, None

    rect = cv2.minAreaRect(largest)
    (rw, rh), angle = rect[1], rect[2]
    corners = np.intp(cv2.boxPoints(rect)) + list(origin)
    longer, shorter = max(rw, rh), min(rw, rh)
    if shorter <= 0.0 or longer / shorter < min_aspect:
        return None, corners
    if rw < rh:
        angle += 90.0
    return angle, corners


def object_axis_angle(image, depth, box, z_ref, depth_scale, depth_tol=0.03):
    """Object long axis in the image. Returns (degrees, corners, source).

    Segmentation is by depth, not intensity. Otsu on the colour crop -- what
    the prototype's calculate_orientation did -- keys on whatever contrast
    happens to be inside the box, and on a narrow crop of a screwdriver it
    latches onto the boundary between shaft and handle and reports an axis
    ~80 degrees off the true one. Depth ignores texture entirely: anything
    within a few centimetres of the object's own median depth is the object.

    Intensity is kept as a fallback for objects the depth sensor cannot see
    (thin, dark, shiny), and the box's own aspect ratio as a last resort, since
    an elongated box already tells you which way the object lies.
    """
    x1, y1, x2, y2 = clamp_box(box, image.shape)
    if x2 <= x1 or y2 <= y1:
        return None, None, 'empty'

    depth_crop = depth[y1:y2, x1:x2].astype(np.float32) * depth_scale
    mask = ((np.abs(depth_crop - z_ref) < depth_tol) & (depth_crop > 0)).astype(np.uint8)
    angle, corners = axis_from_mask(mask, (x1, y1))
    if angle is not None:
        return angle, corners, 'depth'
    if corners is not None:
        # The depth blob is there and it is round. Falling through to
        # intensity would just find another arbitrary angle on the same
        # round thing, and the bbox fallback below would invent one from
        # the box. No axis is the honest answer, and the orchestrator
        # already reads "no axis" as yaw 0 rather than guessing.
        return None, corners, 'round'

    gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    angle, corners = axis_from_mask(thresh, (x1, y1))
    if angle is not None:
        return angle, corners, 'intensity'

    return (0.0 if (x2 - x1) >= (y2 - y1) else -90.0), None, 'bbox'


class Intrinsics(NamedTuple):
    """Pinhole parameters, so the geometry needs no sensor_msgs/CameraInfo.

    Lets the same code run against the driver's CameraInfo, a librealsense
    stream profile, or made-up numbers in a test.
    """

    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_camera_info(cls, info):
        return cls(info.k[0], info.k[4], info.k[2], info.k[5])

    @classmethod
    def from_realsense(cls, profile):
        i = profile.as_video_stream_profile().get_intrinsics()
        return cls(i.fx, i.fy, i.ppx, i.ppy)


def sample_depth(depth, box, depth_scale, min_depth, max_depth):
    """Median depth over the inner half of the box.

    The original prototype read the single centre pixel, so one dropout at the
    object centre sent it down the "depth reading is invalid" path even with
    a perfectly good detection all around it.
    """
    x1, y1, x2, y2 = box
    h, w = depth.shape
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_w = max(2.0, (x2 - x1) * 0.25)
    half_h = max(2.0, (y2 - y1) * 0.25)
    ix1 = max(0, int(cx - half_w))
    ix2 = min(w, int(cx + half_w) + 1)
    iy1 = max(0, int(cy - half_h))
    iy2 = min(h, int(cy + half_h) + 1)
    patch = depth[iy1:iy2, ix1:ix2].astype(np.float32) * depth_scale
    valid = patch[(patch > min_depth) & (patch < max_depth)]
    if valid.size == 0:
        return None, 0
    return float(np.median(valid)), int(valid.size)


def deproject(intrinsics, u, v, z):
    return np.array([(u - intrinsics.cx) * z / intrinsics.fx,
                     (v - intrinsics.cy) * z / intrinsics.fy,
                     z])


def axis_yaw_world(angle_deg, intrinsics, rot, z):
    """Image-plane long axis -> yaw in the world XY plane.

    The camera is pitched 60deg forward, so the image angle is not a world
    yaw. Turn it into a camera-frame direction first (a pixel step maps to
    dx/fx*z, dy/fy*z at constant depth), rotate that into the world, then
    project onto XY.
    """
    a = math.radians(angle_deg)
    d_cam = np.array([math.cos(a) * z / intrinsics.fx,
                      math.sin(a) * z / intrinsics.fy, 0.0])
    n = np.linalg.norm(d_cam)
    if n < 1e-9:
        return None
    d_world = rot @ (d_cam / n)
    if abs(d_world[0]) < 1e-6 and abs(d_world[1]) < 1e-6:
        return None                        # axis points straight up: no yaw
    return math.atan2(d_world[1], d_world[0])


# A box bigger than this fraction of the frame is not an object.
#
# paligemma-3b-pt-224 is a pretrained checkpoint and always answers: asked for
# something that is not in the scene, it returns a box spanning most of the
# view rather than nothing. Observed here -- "detect red battery" on a table
# holding a roll of tape and a spirit level came back as a band across the
# whole table, 98527 depth pixels, centred on nothing in particular. The centre
# of that band then became a grasp target in the middle of the table, out of
# reach of both arms, and the failure looked like a reach problem three steps
# downstream. Cheaper to refuse it here.
MAX_BOX_FRACTION = 0.25


def build_detections(color, depth, raw, intrinsics, rot, trans, depth_scale,
                     min_depth, max_depth, axis_depth_tolerance,
                     max_box_fraction=MAX_BOX_FRACTION):
    """Turn parsed boxes into the records published on /vlm/detections.

    Shared by the node and by vlm_detect.py on purpose: what the standalone
    viewer draws and what the orchestrator acts on come out of the same
    function, so a dot on the image cannot mean something different from the
    coordinate the arm is sent to.

    Returns (detections, overlay_items). Boxes with no usable depth are absent
    from `detections` but present in `overlay_items`, because seeing a box with
    no depth is the whole diagnosis when an object refuses to be picked.
    """
    detections = []
    overlay_items = []
    frame_area = float(color.shape[0] * color.shape[1]) or 1.0
    for det in raw:
        box, center = det['box'], det['center']

        x1, y1, x2, y2 = box
        fraction = abs((x2 - x1) * (y2 - y1)) / frame_area
        if max_box_fraction and fraction > max_box_fraction:
            # Kept on the overlay so the reason is visible rather than the
            # object silently going missing.
            overlay_items.append((None, {
                'bbox_px': [int(v) for v in box],
                'center_px': [int(center[0]), int(center[1])],
                'depth_m': None,
                'rejected': f'box covers {fraction * 100:.0f}% of the frame',
            }))
            continue

        z, n_px = sample_depth(depth, box, depth_scale, min_depth, max_depth)
        if z is None:
            overlay_items.append((None, {
                'bbox_px': [int(v) for v in box],
                'center_px': [int(center[0]), int(center[1])],
                'depth_m': None,
            }))
            continue
        point_cam = deproject(intrinsics, center[0], center[1], z)
        point_world = rot @ point_cam + trans

        angle, corners, axis_source = object_axis_angle(
            color, depth, box, z, depth_scale, axis_depth_tolerance)
        yaw = (axis_yaw_world(angle, intrinsics, rot, z)
               if angle is not None else None)

        record = {
            'bbox_px': [int(v) for v in box],
            'center_px': [int(center[0]), int(center[1])],
            'depth_m': round(z, 4),
            'point_cam': [round(float(v), 4) for v in point_cam],
            'point': [round(float(v), 4) for v in point_world],
            'axis_yaw': None if yaw is None else round(yaw, 4),
            'image_angle_deg': None if angle is None else round(float(angle), 2),
            'axis_source': axis_source,
            'depth_px': n_px,
        }
        detections.append(record)
        overlay_items.append((corners, record))

    # Sort nearest-first so consumers can just take detections[0].
    detections.sort(key=lambda d: d['depth_m'])
    return detections, overlay_items


FONT = cv2.FONT_HERSHEY_SIMPLEX


def put_lines(canvas, lines, x, y, colour, scale=0.42, line_h=15):
    """Text block with a dark backing, clamped to stay inside the canvas."""
    h, w = canvas.shape[:2]
    widest = max((cv2.getTextSize(t, FONT, scale, 1)[0][0] for t in lines),
                 default=0)
    x = max(2, min(x, w - widest - 4))
    y = max(line_h, min(y, h - line_h * len(lines) - 2))

    overlay = canvas.copy()
    cv2.rectangle(overlay, (x - 3, y - line_h + 3),
                  (x + widest + 3, y + line_h * (len(lines) - 1) + 5),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)

    for i, text in enumerate(lines):
        cv2.putText(canvas, text, (x, y + i * line_h), FONT, scale, colour, 1,
                    cv2.LINE_AA)


def colorize_depth(depth, depth_scale, near=0.2, far=2.0):
    """Depth to a viewable rainbow, the way rs.colorizer did in the prototype."""
    metres = depth.astype(np.float32) * depth_scale
    norm = np.clip((metres - near) / max(far - near, 1e-6), 0.0, 1.0)
    view = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    view[metres <= 0] = 0                       # dropouts stay black, not red
    return view


def render_debug(color, depth, items, prompt, hud=(), dashboard=False,
                 depth_scale=0.001):
    """Annotated view of what the detector saw and what it produced.

    `items` is [(corners, record), ...] in detection order, where record is the
    same dict published on /vlm/detections -- so what you read on the image and
    what the orchestrator acts on cannot drift apart.
    """
    canvas = color.copy()
    label = prompt.replace('detect ', '')

    for index, (corners, record) in enumerate(items):
        x1, y1, x2, y2 = record['bbox_px']
        has_depth = record.get('depth_m') is not None
        box_colour = (0, 165, 255) if has_depth else (0, 0, 255)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_colour, 2)
        cv2.circle(canvas, tuple(record['center_px']), 4, (0, 0, 255), -1)
        if corners is not None:
            cv2.drawContours(canvas, [corners], 0, (0, 255, 0), 2)

        if not has_depth:
            why = record.get('rejected') or 'no valid depth'
            put_lines(canvas, [f'#{index} {label}', why],
                      x1, y2 + 16, (0, 0, 255))
            continue

        wx, wy, wz = record['point']
        yaw = record['axis_yaw']
        lines = [
            f'#{index} {label}  d={record["depth_m"]:.3f}m',
            f'xyz {wx:+.3f} {wy:+.3f} {wz:+.3f}',
            (f'yaw {math.degrees(yaw):+.1f}deg' if yaw is not None
             else 'no axis')
            + ('' if record['image_angle_deg'] is None
               else f'  img {record["image_angle_deg"]:+.0f}'),
            f'axis={record["axis_source"]}  depth_px={record["depth_px"]}',
        ]
        # Below the box when there is room, above it otherwise.
        below = y2 + 16
        put_lines(canvas, lines, x1,
                  below if below + 15 * len(lines) < canvas.shape[0] else y1 - 62,
                  (255, 255, 255))

    header = list(hud)
    if not items:
        header.insert(0, f'scanning for "{label}" - nothing detected')
    else:
        header.insert(0, f'target: {label}')
    put_lines(canvas, header, 8, 18, (0, 255, 255), scale=0.45, line_h=17)

    if dashboard:
        # Boxes go on the depth panel too: this is where you see *why* a
        # detection had no depth -- a black hole where the object should be.
        depth_view = colorize_depth(depth, depth_scale)
        for corners, record in items:
            x1, y1, x2, y2 = record['bbox_px']
            cv2.rectangle(depth_view, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.circle(depth_view, tuple(record['center_px']), 4, (0, 0, 0), -1)
            if corners is not None:
                cv2.drawContours(depth_view, [corners], 0, (0, 0, 0), 1)
        canvas = np.hstack((canvas, depth_view))
    return canvas
