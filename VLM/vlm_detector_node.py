#!/usr/bin/env python3
"""PaliGemma object detector as a ROS 2 node, publishing 3D points in `world`.

This is the ROS port of 3d_coordinates.py. Three things changed, all forced by
how this workspace is wired:

1. It does NOT open the RealSense itself. launch_everything.launch.py already
   starts realsense2_camera for the MoveIt octomap, and the D455 can only be
   claimed by one process -- a second pyrealsense2 pipeline just fails. So this
   subscribes to the driver's colour + aligned-depth topics instead.

2. The camera->robot transform comes from tf2, not a hardcoded matrix.
   3d_coordinates.py had T_CAM_TO_ROBOT = (0.150, 0.450, 0.600), but the
   calibrated mount in v10.urdf.xacro is xyz="0.10175 0 0.93272" rpy="0 1.0472 0"
   parented to `world` (see cam_org.txt). tf2 tracks that automatically, so
   re-measuring the mount needs no change here.

3. No cv_bridge. It is built against numpy 1.x and throws an _ARRAY_API ABI
   error in this venv's numpy 2.2.6. Decoding sensor_msgs/Image by hand is three
   lines and avoids the whole problem.

Runs in VLM/.venv (torch 2.14+cu130), which can never share a process with
cuRobo's pinned torch 2.7+cu128 -- hence a separate node talking over DDS.
Launch it via run_vlm_detector.sh, which sanitises the environment for that.

Publishes:
  /vlm/detections       std_msgs/String        JSON, authoritative (see below)
  /vlm/detection_poses  geometry_msgs/PoseArray  same data, for RViz
  /vlm/debug_image      sensor_msgs/Image      annotated colour frame
Subscribes:
  /vlm/prompt           std_msgs/String        retarget at runtime

JSON is used for /vlm/detections because vision_msgs is not installed on this
box and apt-installing it would need root; every other message type here ships
with ros-humble-desktop. Schema:

  {"stamp": <float secs>, "frame_id": "world", "prompt": "detect screwdriver",
   "detections": [{"bbox_px": [x1,y1,x2,y2], "center_px": [cx,cy],
                   "depth_m": 0.62, "point_cam": [x,y,z], "point": [x,y,z],
                   "axis_yaw": 1.23, "image_angle_deg": 12.3,
                   "axis_source": "depth", "depth_px": 420}]}

`axis_source` says how the orientation was found -- "depth" is the good case,
"intensity" and "bbox" are fallbacks worth noticing if grasps start missing.

`axis_yaw` is the object's long axis in the world XY plane. To grasp across it,
the gripper's closing direction must be axis_yaw + 90deg -- pick_place_orchestrator.py
does that.
"""

import json
import math
import re
import threading
import time

import cv2
import numpy as np
import rclpy
import torch
from PIL import Image as PILImage
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

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
    """Unchanged from 3d_coordinates.py: <locNNNN> quadruples -> pixel boxes."""
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


def axis_from_mask(mask, origin, min_area=30):
    """Long axis of the biggest blob in `mask`, in degrees, plus its corners.

    The angle is such that the axis direction is (cos a, sin a) in pixel
    coordinates. minAreaRect's own angle describes the rect's first edge, which
    is the short one whenever width < height, so the +90 puts it back on the
    long axis. Verified against OpenCV 5, whose rect angles run in [-90, 0):
    every orientation comes back correct modulo 180 degrees, which is all an
    undirected grasp axis needs.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None, None

    rect = cv2.minAreaRect(largest)
    (rw, rh), angle = rect[1], rect[2]
    if rw < rh:
        angle += 90.0
    return angle, np.intp(cv2.boxPoints(rect)) + list(origin)


def object_axis_angle(image, depth, box, z_ref, depth_scale, depth_tol=0.03):
    """Object long axis in the image. Returns (degrees, corners, source).

    Segmentation is by depth, not intensity. Otsu on the colour crop -- what
    3d_coordinates.py's calculate_orientation did -- keys on whatever contrast
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

    gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    angle, corners = axis_from_mask(thresh, (x1, y1))
    if angle is not None:
        return angle, corners, 'intensity'

    return (0.0 if (x2 - x1) >= (y2 - y1) else -90.0), None, 'bbox'


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
    """Depth to a viewable rainbow, the way rs.colorizer did in 3d_coordinates.py."""
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
            put_lines(canvas, [f'#{index} {label}', 'no valid depth'],
                      x1, y2 + 16, (0, 0, 255))
            continue

        wx, wy, wz = record['point']
        yaw = record['axis_yaw']
        lines = [
            f'#{index} {label}  d={record["depth_m"]:.3f}m',
            f'xyz {wx:+.3f} {wy:+.3f} {wz:+.3f}',
            (f'yaw {math.degrees(yaw):+.1f}deg' if yaw is not None
             else 'yaw unavailable') + f'  img {record["image_angle_deg"]:+.0f}',
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


class VlmDetectorNode(Node):

    def __init__(self):
        super().__init__('vlm_detector')

        self.declare_parameter('model_id', 'google/paligemma-3b-pt-224')
        self.declare_parameter('prompt', 'detect screwdriver')
        self.declare_parameter('inference_period', 0.4)
        self.declare_parameter('target_frame', 'world')
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic',
                               '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('depth_scale', 0.001)        # 16UC1 millimetres
        self.declare_parameter('min_depth', 0.15)
        self.declare_parameter('max_depth', 3.0)
        self.declare_parameter('max_stamp_skew', 0.08)
        self.declare_parameter('axis_depth_tolerance', 0.03)
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('show_window', False)        # needs a DISPLAY
        self.declare_parameter('debug_dashboard', False)    # colour + depth

        p = self.get_parameter
        self.target_frame = p('target_frame').value
        self.depth_scale = p('depth_scale').value
        self.min_depth = p('min_depth').value
        self.max_depth = p('max_depth').value
        self.max_stamp_skew = p('max_stamp_skew').value
        self.period = p('inference_period').value
        self.prompt = p('prompt').value

        self._lock = threading.Lock()
        self._color = None
        self._depth = None
        self._info = None
        self._warned_distortion = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(Image, p('color_topic').value, self._on_color, 1)
        self.create_subscription(Image, p('depth_topic').value, self._on_depth, 1)
        self.create_subscription(CameraInfo, p('info_topic').value, self._on_info, 1)

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/vlm/prompt', self._on_prompt, latched)

        self.pub_json = self.create_publisher(String, '/vlm/detections', 10)
        self.pub_poses = self.create_publisher(PoseArray, '/vlm/detection_poses', 10)
        self.pub_debug = self.create_publisher(Image, '/vlm/debug_image', 1)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_id = p('model_id').value
        self.get_logger().info(f'loading {model_id} onto {self.device} ...')
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32,
        ).to(self.device)
        self.get_logger().info('model loaded')

        # Inference blocks for a few hundred ms; keeping it off the executor
        # thread means camera callbacks and tf keep flowing while it runs.
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._inference_loop, daemon=True)
        self._worker.start()
        self.get_logger().info(f'detecting: {self.prompt!r}')

    # -- subscriptions -------------------------------------------------------

    def _on_color(self, msg):
        with self._lock:
            self._color = msg

    def _on_depth(self, msg):
        with self._lock:
            self._depth = msg

    def _on_info(self, msg):
        with self._lock:
            self._info = msg
        if not self._warned_distortion:
            self._warned_distortion = True
            worst = max((abs(v) for v in msg.d), default=0.0)
            # Deprojection below uses K only. On the D455's colour sensor these
            # coefficients are near zero; if this logs something large, the
            # few-mm error it implies matters for grasping and should be undone.
            self.get_logger().info(
                f'colour intrinsics {msg.width}x{msg.height} '
                f'fx={msg.k[0]:.1f} fy={msg.k[4]:.1f} '
                f'cx={msg.k[2]:.1f} cy={msg.k[5]:.1f} | max |distortion|={worst:.4f}')

    def _on_prompt(self, msg):
        text = msg.data.strip()
        if not text:
            return
        if not text.lower().startswith('detect'):
            text = f'detect {text}'
        if text != self.prompt:
            self.prompt = text
            self.get_logger().info(f'target changed to {text!r}')

    # -- geometry ------------------------------------------------------------

    def _sample_depth(self, depth, box):
        """Median depth over the inner half of the box.

        3d_coordinates.py read the single centre pixel, so one dropout at the
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
        patch = depth[iy1:iy2, ix1:ix2].astype(np.float32) * self.depth_scale
        valid = patch[(patch > self.min_depth) & (patch < self.max_depth)]
        if valid.size == 0:
            return None, 0
        return float(np.median(valid)), int(valid.size)

    def _deproject(self, info, u, v, z):
        fx, fy = info.k[0], info.k[4]
        cx, cy = info.k[2], info.k[5]
        return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])

    def _lookup_camera_to_target(self, source_frame, stamp):
        tf = self.tf_buffer.lookup_transform(
            self.target_frame, source_frame, stamp,
            timeout=rclpy.duration.Duration(seconds=0.2))
        t = tf.transform.translation
        q = tf.transform.rotation
        return quat_to_rot(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z])

    def _axis_yaw_world(self, angle_deg, info, rot, z):
        """Image-plane long axis -> yaw in the world XY plane.

        The camera is pitched 60deg forward, so the image angle is not a world
        yaw. Turn it into a camera-frame direction first (a pixel step maps to
        dx/fx*z, dy/fy*z at constant depth), rotate that into the world, then
        project onto XY.
        """
        a = math.radians(angle_deg)
        fx, fy = info.k[0], info.k[4]
        d_cam = np.array([math.cos(a) * z / fx, math.sin(a) * z / fy, 0.0])
        n = np.linalg.norm(d_cam)
        if n < 1e-9:
            return None
        d_world = rot @ (d_cam / n)
        if abs(d_world[0]) < 1e-6 and abs(d_world[1]) < 1e-6:
            return None                        # axis points straight up: no yaw
        return math.atan2(d_world[1], d_world[0])

    # -- main loop -----------------------------------------------------------

    def _inference_loop(self):
        while not self._stop.is_set():
            started = time.time()
            try:
                self._step()
            except Exception as exc:                       # keep the node alive
                self.get_logger().warn(f'inference step failed: {exc}')
            remaining = self.period - (time.time() - started)
            if remaining > 0:
                self._stop.wait(remaining)

    def _step(self):
        with self._lock:
            color_msg, depth_msg, info = self._color, self._depth, self._info
        if color_msg is None or depth_msg is None or info is None:
            self.get_logger().info('waiting for colour, aligned depth and camera_info ...',
                                   throttle_duration_sec=5.0)
            return

        def secs(msg):
            return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if abs(secs(color_msg) - secs(depth_msg)) > self.max_stamp_skew:
            self.get_logger().warn('colour and depth stamps disagree; skipping frame',
                                   throttle_duration_sec=5.0)
            return

        color = decode_image(color_msg)
        depth = decode_image(depth_msg)
        if depth.shape[:2] != color.shape[:2]:
            self.get_logger().error(
                f'depth {depth.shape[:2]} != colour {color.shape[:2]}; '
                'align_depth.enable must be true in launch_everything.launch.py',
                throttle_duration_sec=10.0)
            return

        prompt = self.prompt
        height, width = color.shape[:2]

        inference_started = time.time()
        inputs = self.processor(text=prompt,
                                images=PILImage.fromarray(color[:, :, ::-1]),
                                return_tensors='pt').to(self.device)
        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=100)
        text = self.processor.batch_decode(generated, skip_special_tokens=False)[0]
        elapsed = time.time() - inference_started
        raw = parse_paligemma_coordinates(text, width, height)

        try:
            rot, trans = self._lookup_camera_to_target(color_msg.header.frame_id,
                                                       color_msg.header.stamp)
        except Exception as exc:
            self.get_logger().warn(
                f'no transform {color_msg.header.frame_id} -> {self.target_frame}: {exc}',
                throttle_duration_sec=5.0)
            return

        detections = []
        corners_by_index = []
        for det in raw:
            box, center = det['box'], det['center']
            z, n_px = self._sample_depth(depth, box)
            if z is None:
                # Keep it for the overlay: seeing a box with no depth is the
                # whole diagnosis when an object refuses to be picked.
                corners_by_index.append((None, {
                    'bbox_px': [int(v) for v in box],
                    'center_px': [int(center[0]), int(center[1])],
                    'depth_m': None,
                }))
                continue
            point_cam = self._deproject(info, center[0], center[1], z)
            point_world = rot @ point_cam + trans

            angle, corners, axis_source = object_axis_angle(
                color, depth, box, z, self.depth_scale,
                self.get_parameter('axis_depth_tolerance').value)
            axis_yaw = (self._axis_yaw_world(angle, info, rot, z)
                        if angle is not None else None)

            record = {
                'bbox_px': [int(v) for v in box],
                'center_px': [int(center[0]), int(center[1])],
                'depth_m': round(z, 4),
                'point_cam': [round(float(v), 4) for v in point_cam],
                'point': [round(float(v), 4) for v in point_world],
                'axis_yaw': None if axis_yaw is None else round(axis_yaw, 4),
                'image_angle_deg': None if angle is None else round(float(angle), 2),
                'axis_source': axis_source,
                'depth_px': n_px,
            }
            detections.append(record)
            corners_by_index.append((corners, record))

        # Sort nearest-first so consumers can just take detections[0].
        detections.sort(key=lambda d: d['depth_m'])

        now = self.get_clock().now()
        payload = {
            'stamp': now.nanoseconds * 1e-9,
            'frame_id': self.target_frame,
            'prompt': prompt,
            'detections': detections,
        }
        self.pub_json.publish(String(data=json.dumps(payload)))

        poses = PoseArray()
        poses.header.stamp = now.to_msg()
        poses.header.frame_id = self.target_frame
        for d in detections:
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = d['point']
            qx, qy, qz, qw = yaw_to_quat(d['axis_yaw'] or 0.0)
            pose.orientation.x, pose.orientation.y = qx, qy
            pose.orientation.z, pose.orientation.w = qz, qw
            poses.poses.append(pose)
        self.pub_poses.publish(poses)

        if self.get_parameter('publish_debug_image').value or \
                self.get_parameter('show_window').value:
            canvas = render_debug(
                color, depth, corners_by_index, prompt,
                hud=[f'{width}x{height}  {1.0 / max(elapsed, 1e-3):.1f} Hz'
                     f'  {len(detections)}/{len(raw)} with depth',
                     f'frame: {self.target_frame}'],
                dashboard=self.get_parameter('debug_dashboard').value,
                depth_scale=self.depth_scale)
            self._publish_debug(canvas, color_msg.header.stamp)

    def _publish_debug(self, canvas, stamp):
        if self.get_parameter('publish_debug_image').value:
            msg = Image()
            msg.header.stamp = stamp
            msg.header.frame_id = 'camera_color_optical_frame'
            msg.height, msg.width = canvas.shape[:2]
            msg.encoding = 'bgr8'
            msg.is_bigendian = 0
            msg.step = canvas.shape[1] * 3
            msg.data = canvas.tobytes()
            self.pub_debug.publish(msg)

        if self.get_parameter('show_window').value:
            cv2.imshow('VLM detector', canvas)
            cv2.waitKey(1)

    def destroy_node(self):
        self._stop.set()
        self._worker.join(timeout=5.0)
        return super().destroy_node()


def main():
    rclpy.init()
    node = VlmDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
