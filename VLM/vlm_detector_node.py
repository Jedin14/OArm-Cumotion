#!/usr/bin/env python3
"""PaliGemma object detector as a ROS 2 node, publishing 3D points in `world`.

This is the ROS port of the original standalone prototype. Three things
changed, all forced by how this workspace is wired:

1. It does NOT open the RealSense itself. launch_everything.launch.py already
   starts realsense2_camera for the MoveIt octomap, and the D455 can only be
   claimed by one process -- a second pyrealsense2 pipeline just fails. So this
   subscribes to the driver's colour + aligned-depth topics instead.

2. The camera->robot transform comes from tf2, not a hardcoded matrix.
   The prototype had T_CAM_TO_ROBOT = (0.150, 0.450, 0.600), but the
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
   "image_size": [width, height],
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
import threading
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

# torch, transformers and PIL are imported inside _load_model, not here.
# They cost seconds and gigabytes, and importing this module for its
# re-exported geometry should not pay that -- nor should it fail on a python
# whose PIL is too old for transformers, which is what
# "module 'PIL.Image' has no attribute 'Resampling'" was.

# Geometry, parsing and the overlay live in vlm_geometry so they can be
# imported without rclpy, torch or transformers -- see that module.
# Re-exported here because both this node and vlm_detect.py import them.
from vlm_geometry import (  # noqa: F401
    FONT,
    Intrinsics,
    LOC_PATTERN,
    axis_from_mask,
    axis_yaw_world,
    build_detections,
    clamp_box,
    colorize_depth,
    decode_image,
    deproject,
    object_axis_angle,
    parse_paligemma_coordinates,
    put_lines,
    quat_to_rot,
    render_debug,
    sample_depth,
    yaw_to_quat,
)


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
        # When to run at all.
        #
        # 'continuous' is what this used to do unconditionally: a full
        # model.generate every inference_period, for ever, whether or not
        # anybody had asked for anything. Measured on the robot -- 6.4 GB of
        # VRAM held permanently and the GPU busy between cycles, on a machine
        # where cuMotion and anything else have to fit alongside.
        #
        # 'on_demand' runs only after a prompt arrives, for active_window
        # seconds, which is what a pick actually needs: the orchestrator
        # publishes the prompt and then waits for one fresh detection.
        self.declare_parameter('inference_mode', 'on_demand')
        self.declare_parameter('active_window', 6.0)
        # Seconds of idleness after which the weights are moved off the GPU
        # and the cache emptied. 0 keeps them resident, which costs 6.4 GB and
        # saves the second or so it takes to move them back.
        self.declare_parameter('idle_unload_after', 20.0)
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
        self.mode = p('inference_mode').value
        self.active_window = p('active_window').value
        self.idle_unload_after = p('idle_unload_after').value
        # Until when inference is wanted. Started in the past so an on-demand
        # node comes up idle: nothing has asked for anything yet.
        self._wanted_until = 0.0
        self._idle_since = time.time()
        self._on_gpu = True

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

        self._load_model(p('model_id').value)

        # Inference blocks for a few hundred ms; keeping it off the executor
        # thread means camera callbacks and tf keep flowing while it runs.
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._inference_loop, daemon=True)
        self._worker.start()
        self.get_logger().info(f'detecting: {self.prompt!r}')

    def _load_model(self, model_id):
        """Import and load PaliGemma. The heavy dependencies live here only."""
        import torch
        from PIL import Image as PILImage
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        self.torch = torch
        self.pil_image = PILImage
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'loading {model_id} onto {self.device} ...')
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32,
        ).to(self.device)
        self.get_logger().info('model loaded')

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
        # Every prompt is a request, including a repeat of the one already
        # set. The orchestrator republishes the same text each time it wants
        # a look -- at LOCATE, and again to verify the grasp and the place --
        # so treating an unchanged prompt as nothing to do would leave those
        # waiting for a frame that never comes.
        self._wanted_until = time.time() + self.active_window

    # -- geometry ------------------------------------------------------------

    def _lookup_camera_to_target(self, source_frame, stamp):
        tf = self.tf_buffer.lookup_transform(
            self.target_frame, source_frame, stamp,
            timeout=rclpy.duration.Duration(seconds=0.2))
        t = tf.transform.translation
        q = tf.transform.rotation
        return quat_to_rot(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z])

    # -- main loop -----------------------------------------------------------

    def _inference_loop(self):
        while not self._stop.is_set():
            started = time.time()
            if self._wanted():
                self._ensure_loaded()
                try:
                    self._step()
                except Exception as exc:                   # keep the node alive
                    self.get_logger().warn(f'inference step failed: {exc}')
                self._idle_since = time.time()
            else:
                self._maybe_unload()
            remaining = self.period - (time.time() - started)
            if remaining > 0:
                self._stop.wait(remaining)

    def _wanted(self):
        """Is anybody waiting on a detection right now?"""
        if self.mode != 'on_demand':
            return True
        return time.time() < self._wanted_until

    def _ensure_loaded(self):
        """Put the weights back on the GPU if they were parked."""
        # _load_model runs before the worker starts, so this is belt and
        # braces -- but the loop must not be the thing that crashes if that
        # order ever changes.
        if self._on_gpu or getattr(self, 'model', None) is None:
            return
        moved = time.time()
        self.model.to(self.device)
        self._on_gpu = True
        self.get_logger().info(
            f'weights back on {self.device} in {time.time() - moved:.1f} s')

    def _maybe_unload(self):
        """Give the GPU back after a spell with nothing asked of it.

        The weights go to host memory rather than being dropped, so coming
        back costs a transfer rather than a reload from disk -- about a second
        against tens. Worth it: 6.4 GB is most of a 16 GB card, and cuMotion,
        the octomap and anything else have to live in what is left.
        """
        after = self.idle_unload_after
        if not self._on_gpu or after <= 0 or getattr(self, 'device', '') != 'cuda':
            return
        if getattr(self, 'model', None) is None:
            return
        if time.time() - self._idle_since < after:
            return
        self.model.to('cpu')
        self._on_gpu = False
        try:
            self.torch.cuda.empty_cache()
        except Exception:                                  # noqa: BLE001
            pass
        self.get_logger().info(
            f'idle for {after:.0f} s, so the weights are off the GPU. The '
            f'next prompt brings them back, about a second later.')

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
                                images=self.pil_image.fromarray(
                                    color[:, :, ::-1]),
                                return_tensors='pt').to(self.device)
        with self.torch.no_grad():
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

        detections, corners_by_index = build_detections(
            color, depth, raw, Intrinsics.from_camera_info(info), rot, trans,
            self.depth_scale, self.min_depth, self.max_depth,
            self.get_parameter('axis_depth_tolerance').value)

        now = self.get_clock().now()
        payload = {
            'stamp': now.nanoseconds * 1e-9,
            'frame_id': self.target_frame,
            'prompt': prompt,
            # Needed to say which half of the camera view a detection is in,
            # which is how the orchestrator picks an arm. center_px on its own
            # cannot answer that without knowing the frame width.
            'image_size': [width, height],
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
