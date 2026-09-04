#!/usr/bin/env python3
"""See what the VLM sees: a live view with a dot on the object and its coordinates.

The detector node publishes an annotated frame on /vlm/debug_image, but reading
that needs RViz or rqt and the whole stack up. This is the same annotation as a
standalone tool -- point it at a prompt and it shows you, per frame, where the
model thinks the object is and what coordinate it hands the arm.

    # live window, camera owned by the running stack
    VLM/run_in_vlm_env.sh vlm_detect.py "screwdriver"

    # one frame, annotated PNG on disk, coordinates on stdout
    VLM/run_in_vlm_env.sh vlm_detect.py "screwdriver" --once

    # stack not running: open the camera directly
    VLM/run_in_vlm_env.sh vlm_detect.py "red battery" --source realsense

Two sources, because the D455 can only be claimed by one process:

  ros         subscribe to realsense2_camera's colour and aligned-depth topics.
              Use this while the robot is up -- the driver owns the device, and
              a second pyrealsense2 pipeline simply fails. World coordinates
              come from tf2, exactly as the node gets them.
  realsense   open the camera here. Only when the stack is down. There is no
              tf2 then, so coordinates are reported in the camera optical frame
              unless --mount is given.

Default is "auto": use ros if the colour topic is being published, else the
camera directly.

The detection and geometry come from vlm_detector_node.build_detections -- the
same function the node uses -- so the dot you see and the coordinate the
orchestrator acts on cannot mean different things. The prompt is normalised the
same way too: "screwdriver", "the screwdriver" and "pick up the screwdriver"
all become "detect screwdriver", because paligemma-3b-pt-224 is a pretrained
checkpoint and "detect X" is the well-formed prompt shape.
"""

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np

WRAPPER = 'VLM/run_in_vlm_env.sh vlm_detect.py'


def require(module, needed_for):
    """Import a heavy dependency, or explain which Python to use.

    PaliGemma and the RealSense bindings live in VLM/.venv, and native
    setup.bash exports a PYTHONPATH that beats a venv's site-packages -- so
    this has to run through the wrapper. A bare ModuleNotFoundError three
    frames deep does not say that.
    """
    try:
        return __import__(module)
    except ImportError as exc:
        print(f'error: {module} is not available, and it is needed {needed_for}.'
              f'\n       It lives in VLM/.venv. Run this through the wrapper, '
              f'which puts that venv\n       on the path and scrubs the cuRobo '
              f'overlay:\n'
              f'\n           {WRAPPER} ...\n'
              f'\n       ({type(exc).__name__}: {exc})', file=sys.stderr)
        raise SystemExit(2)


WS = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if WS not in sys.path:
    sys.path.insert(0, WS)
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from vlm_geometry import (                 # noqa: E402
    Intrinsics,
    build_detections,
    decode_image,
    parse_paligemma_coordinates,
    quat_to_rot,
    render_debug,
)

try:
    from vlm_prompt import to_detection_prompt
except ImportError:                        # standalone checkout
    def to_detection_prompt(text):
        cleaned = (text or '').strip()
        return f'detect {cleaned}' if cleaned else ''

COLOR_TOPIC = '/camera/camera/color/image_raw'
DEPTH_TOPIC = '/camera/camera/aligned_depth_to_color/image_raw'
INFO_TOPIC = '/camera/camera/color/camera_info'

# The calibrated mount from v10.urdf.xacro: xyz="0.10175 0 0.93272"
# rpy="0 1.0472 0", parented to world. Only used by --source realsense, where
# there is no tf2 to ask. If the mount is re-measured, the URDF is the truth
# and this is a copy -- which is why it is opt-in via --mount rather than the
# default.
URDF_MOUNT_XYZ = (0.10175, 0.0, 0.93272)
URDF_MOUNT_RPY = (0.0, 1.0472, 0.0)

# realsense2_camera publishes optical frames; a bare pyrealsense2 pipeline does
# not apply the ROS optical convention, so --source realsense needs it here.
# Optical: +x right, +y down, +z forward. ROS body: +x forward, +y left, +z up.
OPTICAL_FROM_BODY = np.array([[0.0, 0.0, 1.0],
                              [-1.0, 0.0, 0.0],
                              [0.0, -1.0, 0.0]])


def rpy_to_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]])


class Model:
    """PaliGemma, loaded once."""

    def __init__(self, model_id):
        torch = require('torch', 'to run the model')
        require('transformers', 'to load PaliGemma')
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
        self.torch = torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'loading {model_id} onto {self.device} '
              f'(tens of seconds, several GB of VRAM) ...', flush=True)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32,
        ).to(self.device)
        print('model loaded', flush=True)

    def boxes(self, color, prompt):
        """Parsed boxes for this frame, plus how long inference took."""
        from PIL import Image as PILImage
        started = time.time()
        inputs = self.processor(
            text=prompt,
            images=PILImage.fromarray(color[:, :, ::-1]),
            return_tensors='pt').to(self.device)
        with self.torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=100)
        text = self.processor.batch_decode(generated,
                                           skip_special_tokens=False)[0]
        height, width = color.shape[:2]
        return parse_paligemma_coordinates(text, width, height), \
            time.time() - started


class RosSource:
    """Colour, aligned depth and intrinsics off the running driver, plus tf2."""

    name = 'ros'

    def __init__(self, target_frame='world'):
        require('rclpy', 'to read the camera topics')
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import CameraInfo, Image
        from tf2_ros import Buffer, TransformListener

        self.rclpy = rclpy
        rclpy.init()
        self.node = Node('vlm_detect')
        self.target_frame = target_frame
        self._color = self._depth = self._info = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)

        def keep(attr):
            def cb(msg):
                setattr(self, attr, msg)
            return cb

        self.node.create_subscription(Image, COLOR_TOPIC, keep('_color'), 1)
        self.node.create_subscription(Image, DEPTH_TOPIC, keep('_depth'), 1)
        self.node.create_subscription(CameraInfo, INFO_TOPIC, keep('_info'), 1)

    @staticmethod
    def topic_is_live(timeout=3.0):
        """Is the driver publishing colour? Decides "auto"."""
        try:
            import rclpy
        except ImportError:
            return False
        from rclpy.node import Node
        from sensor_msgs.msg import Image
        rclpy.init()
        try:
            probe = Node('vlm_detect_probe')
            seen = []
            probe.create_subscription(Image, COLOR_TOPIC,
                                      lambda m: seen.append(True), 1)
            deadline = time.time() + timeout
            while time.time() < deadline and not seen:
                rclpy.spin_once(probe, timeout_sec=0.1)
            probe.destroy_node()
            return bool(seen)
        finally:
            if rclpy.ok():
                rclpy.shutdown()

    def frame(self, timeout=10.0):
        """(color, depth, intrinsics, rot, trans, note) or None."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.1)
            if self._color is not None and self._depth is not None \
                    and self._info is not None:
                break
        if self._color is None:
            return None, f'no frames on {COLOR_TOPIC}'
        if self._depth is None:
            # Worth saying plainly: this exact failure reported an object four
            # metres away and below the floor.
            return None, (f'colour is arriving but {DEPTH_TOPIC} is not. '
                          'Every coordinate depends on depth, so there is '
                          'nothing to report. Check:  ros2 topic hz '
                          f'{DEPTH_TOPIC}')
        if self._info is None:
            return None, f'no camera_info on {INFO_TOPIC}'

        color = decode_image(self._color)
        depth = decode_image(self._depth)
        if depth.shape[:2] != color.shape[:2]:
            return None, (f'depth {depth.shape[:2]} != colour '
                          f'{color.shape[:2]} -- align_depth.enable must be '
                          'true in launch_everything.launch.py')

        rot = np.eye(3)
        trans = np.zeros(3)
        note = f'{self.target_frame} (tf2)'
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame, self._color.header.frame_id,
                self.rclpy.time.Time())
            t, q = tf.transform.translation, tf.transform.rotation
            rot = quat_to_rot(q.x, q.y, q.z, q.w)
            trans = np.array([t.x, t.y, t.z])
        except Exception as exc:                 # noqa: BLE001 - reported
            note = (f'camera optical frame -- no transform to '
                    f'{self.target_frame} ({type(exc).__name__})')
        return (color, depth, Intrinsics.from_camera_info(self._info),
                rot, trans, note), None

    def close(self):
        self.node.destroy_node()
        if self.rclpy.ok():
            self.rclpy.shutdown()


class RealsenseSource:
    """The camera opened here. Only valid when the stack is not running."""

    name = 'realsense'

    def __init__(self, width=848, height=480, fps=15, mount=False, warmup=5,
                 reset=True):
        rs = require('pyrealsense2', 'to open the camera directly')
        self.rs = rs
        self.mount = mount
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height,
                                  rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height,
                                  rs.format.z16, fps)
        self.align = rs.align(rs.stream.color)
        print(f'opening the camera at {width}x{height}@{fps} ...', flush=True)

        settled = self._start(warmup)
        if settled == 0 and reset:
            # A D455 can enumerate, accept a pipeline start, and then deliver
            # nothing at all -- confirmed here with a bare librealsense
            # pipeline, so it is the device and not this code. A hardware reset
            # clears it: 0/10 frames before, 10/10 after. Worth doing
            # automatically, because the symptom otherwise looks like the wrong
            # resolution, a busy device, or a bad cable.
            print('no frames from the camera; hardware-resetting it',
                  flush=True)
            self._hardware_reset()
            settled = self._start(warmup)
        if settled == 0:
            raise SystemExit(
                'error: the camera is open but delivers no frames, and a '
                'hardware reset did not help.\n'
                '       Replug it, and check it is on a USB 3 port:  '
                'rs-enumerate-devices')
        print(f'camera open, depth scale {self.depth_scale}, '
              f'{settled} warm-up frames', flush=True)

    def _start(self, warmup):
        """Start the pipeline and read warm-up frames. Returns how many came.

        Auto-exposure needs a few frames to settle and the first ones after a
        start are routinely missing, so these are read and discarded -- which
        doubles as the test for whether the device is delivering at all.
        """
        self.pipeline = self.rs.pipeline()
        self.profile = self.pipeline.start(self.config)
        self.depth_scale = (self.profile.get_device().first_depth_sensor()
                            .get_depth_scale())
        settled = 0
        for _ in range(warmup):
            try:
                self.pipeline.wait_for_frames(2000)
                settled += 1
            except RuntimeError:
                break
        if settled == 0:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass
        return settled

    def _hardware_reset(self, timeout=30.0):
        devices = self.rs.context().query_devices()
        if not len(devices):
            raise SystemExit('error: no RealSense device to reset')
        devices[0].hardware_reset()
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(1.0)
            if len(self.rs.context().query_devices()):
                time.sleep(2.0)          # settle after re-enumeration
                return
        raise SystemExit('error: the camera did not come back after a reset')

    def frame(self, timeout=10.0, tries=3):
        frames = None
        for attempt in range(1, tries + 1):
            try:
                frames = self.pipeline.wait_for_frames(int(timeout * 1000))
                break
            except RuntimeError as exc:
                if attempt == tries:
                    return None, (
                        f'the camera delivered no frames in '
                        f'{timeout * tries:.0f} s ({exc}). It is open, so this '
                        f'is usually the USB link or another process holding '
                        f'the device -- check with:  rs-enumerate-devices')
        frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, 'camera returned an incomplete frame set'
        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        intrinsics = Intrinsics.from_realsense(color_frame.profile)

        if self.mount:
            # Body-frame mount from the URDF, then the optical convention the
            # ROS driver would have applied.
            rot = rpy_to_matrix(*URDF_MOUNT_RPY) @ OPTICAL_FROM_BODY
            trans = np.array(URDF_MOUNT_XYZ)
            note = 'world (URDF mount, NOT tf2 -- verify against the URDF)'
        else:
            rot, trans = np.eye(3), np.zeros(3)
            note = 'camera optical frame (+x right, +y down, +z forward)'
        return (color, depth, intrinsics, rot, trans, note), None

    def close(self):
        self.pipeline.stop()


def describe(detections, frame_note):
    """The coordinates, as text, for a terminal."""
    if not detections:
        return ['nothing detected']
    lines = [f'coordinates in {frame_note}:']
    for index, d in enumerate(detections):
        px, py = d['center_px']
        x, y, z = d['point']
        cx, cy, cz = d['point_cam']
        yaw = d['axis_yaw']
        lines.append(
            f'  #{index}  pixel ({px}, {py})  depth {d["depth_m"]:.3f} m '
            f'from {d["depth_px"]} px')
        lines.append(f'        camera xyz  {cx:+.4f} {cy:+.4f} {cz:+.4f}')
        lines.append(f'        target xyz  {x:+.4f} {y:+.4f} {z:+.4f}')
        axis = (f'{math.degrees(yaw):+.1f} deg' if yaw is not None
                else 'unavailable')
        lines.append(f'        axis yaw    {axis} (from {d["axis_source"]})')
    return lines


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('prompt', nargs='+',
                        help='what to look for, e.g. "screwdriver". Several '
                             'may be given and each is tried on the same '
                             'frame -- the model load is the slow part, so '
                             'finding a word the model responds to costs one '
                             'run, not one per guess.')
    parser.add_argument('--source', default='auto',
                        choices=('auto', 'ros', 'realsense'))
    parser.add_argument('--once', action='store_true',
                        help='one frame, save the annotated image, print, exit')
    parser.add_argument('--save', default=None,
                        help='write the annotated frame here (implies --once '
                             'unless --window)')
    parser.add_argument('--window', action='store_true',
                        help='show a live window (needs DISPLAY)')
    parser.add_argument('--json', default=None,
                        help='also write the detections as JSON here')
    parser.add_argument('--frame', default='world',
                        help='target frame for --source ros (default: world)')
    parser.add_argument('--mount', action='store_true',
                        help='--source realsense: apply the URDF camera mount '
                             'to report world coordinates instead of camera')
    parser.add_argument('--no-reset', action='store_false', dest='reset',
                        help='--source realsense: do not hardware-reset the '
                             'camera when it delivers no frames')
    parser.add_argument('--model', default='google/paligemma-3b-pt-224')
    parser.add_argument('--depth-scale', type=float, default=None,
                        help='metres per depth unit (default: 0.001 for ros, '
                             'the device value for realsense)')
    parser.add_argument('--min-depth', type=float, default=0.15)
    parser.add_argument('--max-depth', type=float, default=3.0)
    parser.add_argument('--axis-depth-tolerance', type=float, default=0.03)
    parser.add_argument('--period', type=float, default=0.4,
                        help='seconds between inferences in window mode')
    args = parser.parse_args()

    prompts = [to_detection_prompt(p) for p in args.prompt]
    prompts = [p for p in prompts if p]
    if not prompts:
        print('error: empty prompt', file=sys.stderr)
        return 2
    print('prompts: ' + ', '.join(repr(p) for p in prompts))

    once = args.once or (args.save is not None and not args.window)
    if not once and not args.window:
        args.window = True
    if args.window and not os.environ.get('DISPLAY'):
        print('note: no DISPLAY, so falling back to a single saved frame')
        args.window, once = False, True

    source_kind = args.source
    if source_kind == 'auto':
        live = RosSource.topic_is_live()
        source_kind = 'ros' if live else 'realsense'
        print(f'source: {source_kind} '
              f'({"colour topic is publishing" if live else "no colour topic"})')

    # The model first, deliberately. Opening the camera and then blocking for
    # the tens of seconds PaliGemma takes leaves librealsense streaming into a
    # queue nobody is reading, and the first wait_for_frames comes back
    # "Frame didn't arrive within 10000". Load the slow thing, then open the
    # device and start reading it straight away.
    model = Model(args.model)

    if source_kind == 'ros':
        source = RosSource(args.frame)
        depth_scale = args.depth_scale if args.depth_scale else 0.001
    else:
        source = RealsenseSource(mount=args.mount, reset=args.reset)
        depth_scale = args.depth_scale if args.depth_scale \
            else source.depth_scale
    saved_any = False
    try:
        while True:
            got, problem = source.frame()
            if got is None:
                print(f'error: {problem}', file=sys.stderr)
                return 1
            color, depth, intrinsics, rot, trans, frame_note = got

            # Every prompt is run against the same frame, so the results are
            # comparable and the model is loaded once.
            best = None
            for prompt in prompts:
                raw, elapsed = model.boxes(color, prompt)
                detections, overlay = build_detections(
                    color, depth, raw, intrinsics, rot, trans, depth_scale,
                    args.min_depth, args.max_depth, args.axis_depth_tolerance)

                print(f'{prompt!r}:')
                for line in describe(detections, frame_note):
                    print('  ' + line)
                if raw and not detections:
                    reasons = [item[1].get('rejected') or 'no usable depth'
                               for item in overlay]
                    for reason in sorted(set(reasons)):
                        print(f'    ({reasons.count(reason)} box(es) '
                              f'discarded: {reason})')
                    if any('covers' in r for r in reasons):
                        print('    A box that size is the model shrugging -- '
                              'paligemma-3b-pt-224 always answers, and '
                              'returns most of\n    the view when what you '
                              'asked for is not in the scene.')
                print(flush=True)
                if best is None or len(detections) > len(best[1]):
                    best = (prompt, detections, overlay, elapsed, len(raw))

            prompt, detections, overlay, elapsed, n_raw = best
            canvas = render_debug(
                color, depth, overlay, prompt,
                hud=[f'{color.shape[1]}x{color.shape[0]}  '
                     f'{1.0 / max(elapsed, 1e-3):.1f} Hz  '
                     f'{len(detections)}/{n_raw} with depth',
                     f'frame: {frame_note}',
                     f'source: {source.name}'],
                depth_scale=depth_scale)

            if args.save or once:
                path = args.save or 'vlm_detection.png'
                cv2.imwrite(path, canvas)
                print(f'wrote {os.path.abspath(path)}')
                saved_any = True
            if args.json:
                with open(args.json, 'w') as handle:
                    json.dump({'prompt': prompt, 'frame': frame_note,
                               'detections': detections}, handle, indent=2)
                print(f'wrote {os.path.abspath(args.json)}')

            if once:
                return 0
            cv2.imshow('vlm_detect', canvas)
            if cv2.waitKey(max(1, int(args.period * 1000))) in (27, ord('q')):
                return 0
    except KeyboardInterrupt:
        print('\nstopped')
        return 0
    finally:
        source.close()
        if args.window and not saved_any:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main())
