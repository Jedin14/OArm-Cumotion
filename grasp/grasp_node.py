#!/usr/bin/env python3
"""GraspNet-baseline as a ROS 2 node: ranked 6-DoF grasps for what the VLM found.

The detector says *which* object and roughly where. This says *how to grab it*
-- a list of scored grasp poses read off the point cloud, with the gripper
opening each one needs. The orchestrator takes the list and the pre-flight
picks the first that flies.

Why this exists, in terms of what was going wrong without it:

  * the grasp height came from one depth pixel plus a fixed offset. Two
    objects on the same table gave points 19 mm apart, and the jaws closed
    11 mm above a screwdriver.
  * the grasp yaw came from an axis estimate over a 2D box. For a roll of
    tape, 8 of 36 orientations were flyable and it insisted on one.
  * nothing predicted the gripper opening, so "closed on nothing" and "held
    something thin" were the same observation.

All three are properties of the object's geometry, and this reads them off it.

Runs in its own environment -- third_party/grasp_venv, with its own torch and
the compiled pointnet2/knn extensions -- for the same reason the detector
does: the versions do not co-exist with cuRobo's. Start it through
grasp/run_grasp_server.sh, which is what puts that environment on the path.

    /grasp/candidates   ranked grasps, JSON on a String, world frame
    /grasp/request      ask for a fresh set (the object's world point as JSON)

Licence: graspnet-baseline is SJTU's, academic/non-profit non-commercial
research use only. See third_party/graspnet-baseline/LICENSE. Nothing here
redistributes it -- the clone and the weights are fetched by
grasp/fetch_graspnet.sh and are .gitignored.
"""

import json
import os
import sys
import threading
import time

import numpy as np

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRASPNET = os.path.join(WS, 'third_party', 'graspnet-baseline')


# graspnet-baseline's own finger geometry, from
# utils/collision_detector.py -- the space between the jaws runs from
# depth - FINGER_LENGTH to depth along the approach axis. Taken from there
# rather than guessed so that what is measured is the volume its own
# collision check reasons about.
FINGER_LENGTH = 0.06

# Below this, the "object" between the jaws is depth noise and its extent
# means nothing.
MIN_SPAN_POINTS = 10


def quat_to_rot(x, y, z, w):
    """Rotation matrix from a quaternion, as three columns."""
    n = (x * x + y * y + z * z + w * w) ** 0.5
    if n == 0.0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rot_to_quat(matrix):
    """(x, y, z, w) from a rotation matrix, by the branch that stays stable."""
    m = matrix
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return ((m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s,
                (m[1, 0] - m[0, 1]) * s, 0.25 / s)
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        return (0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s,
                (m[2, 1] - m[1, 2]) / s)
    if m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        return ((m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s,
                (m[0, 2] - m[2, 0]) / s)
    s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
    return ((m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s,
            (m[1, 0] - m[0, 1]) / s)


def decode_image(msg):
    """A ROS image as a numpy array. Only the encodings the D455 emits."""
    if msg.encoding in ('16UC1', 'mono16'):
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(
            msg.height, msg.width)
    channels = max(1, msg.step // max(1, msg.width))
    array = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, channels)
    if msg.encoding.startswith('rgb'):
        array = array[:, :, ::-1]
    return array


class GraspServer(Node):
    """One inference per request, on the frame the detector last decided from.

    Deliberately request-driven rather than continuous, like the detector: the
    weights are hundreds of megabytes on a card that cuMotion and PaliGemma
    also live on, and a grasp is only wanted at the moment a pick starts.
    """

    def __init__(self):
        super().__init__('grasp_server')
        self.cb = ReentrantCallbackGroup()
        self.lock = threading.Lock()

        self.declare_parameter('checkpoint', os.path.join(
            WS, 'third_party', 'weights', 'checkpoint-rs.tar'))
        self.declare_parameter('num_point', 20000)
        self.declare_parameter('num_view', 300)
        # The workspace, in the camera frame's own metres. Points outside it
        # are dropped before the network sees them: the far wall and the floor
        # are most of the cloud and none of the grasps.
        self.declare_parameter('min_depth', 0.20)
        self.declare_parameter('max_depth', 1.20)
        self.declare_parameter('depth_scale', 0.001)     # 16UC1 millimetres
        # How near a grasp must be to the object the detector named, metres.
        # GraspNet proposes for the whole scene; this is what makes the answer
        # about the thing that was asked for.
        self.declare_parameter('object_radius', 0.06)
        self.declare_parameter('max_candidates', 20)
        self.declare_parameter('min_score', 0.05)
        # What the jaws can actually span, metres, plus whatever margin the
        # grasp should keep. GraspNet was trained on the GraspNet-1Billion
        # gripper, which opens to 100 mm; this one opens to 44 mm, so a good
        # share of what it proposes is a grasp this robot cannot make.
        # Measured on the first real run -- a roll of tape -- every one of
        # the three candidates came back needing 55, 66 and 98 mm.
        #
        # Filtered here rather than left to the orchestrator: a candidate the
        # gripper cannot close on is not a ranked alternative, it is noise,
        # and it would push a usable one off the end of max_candidates.
        self.declare_parameter('gripper_open', 0.044)
        # Room left between the jaws and the object at the commanded width.
        # 0 accepts a grasp that needs the gripper at full stretch.
        self.declare_parameter('width_margin', 0.004)
        # Whether a grasp wider than the jaws is refused at all.
        #
        # On, a grasp the model asks more opening for than the hardware has
        # is measured against the cloud and kept only if the object between
        # the jaws is genuinely narrower. That is the careful reading, and
        # it threw away most of what the model proposed: measured, 'near
        # object widths ran 87-87 mm' against a 40 mm span.
        #
        # Off, the width the model asked for is treated as an upper bound
        # rather than a requirement, up to span_ceiling, and the opening
        # actually commanded is clamped to what the jaws have. So 0-44 mm
        # passes through as it does now and 44-100 mm is commanded as 44.
        # The jaws close until they meet the object either way -- they are
        # a position command against a torque cap, not a caliper -- so the
        # cost of being wrong is a grasp that closes on nothing, which
        # verify_grasp already catches.
        self.declare_parameter('span_check', True)
        self.declare_parameter('span_ceiling', 0.100)
        self.declare_parameter('collision_thresh', 0.01)
        self.declare_parameter('voxel_size', 0.01)
        self.declare_parameter('target_frame', 'world')
        self.declare_parameter('color_topic',
                               '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic',
                               '/camera/camera/aligned_depth_to_color/'
                               'image_raw')
        self.declare_parameter('info_topic',
                               '/camera/camera/color/camera_info')
        self.declare_parameter('idle_unload_after', 30.0)

        p = self.get_parameter
        self.target_frame = p('target_frame').value
        self.depth_scale = p('depth_scale').value

        self._color = None
        self._depth = None
        self._info = None
        self._pending = None
        self._idle_since = time.time()
        self._on_gpu = True

        latched = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Image, p('color_topic').value,
                                 self._on_color, 1, callback_group=self.cb)
        self.create_subscription(Image, p('depth_topic').value,
                                 self._on_depth, 1, callback_group=self.cb)
        self.create_subscription(CameraInfo, p('info_topic').value,
                                 self._on_info, 1, callback_group=self.cb)
        # Either trigger works: an explicit request naming the object's world
        # point, or the detector's own output. The second means a pick gets
        # grasps without the orchestrator having to ask twice.
        self.create_subscription(String, '/grasp/request',
                                 self._on_request, 10, callback_group=self.cb)
        self.create_subscription(String, '/vlm/detections',
                                 self._on_detections, 10,
                                 callback_group=self.cb)
        self.candidates_pub = self.create_publisher(
            String, '/grasp/candidates', latched)

        self.tf_buffer = Buffer()
        TransformListener(self.tf_buffer, self)

        self._load()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()
        self.get_logger().info('ready; waiting for a request')

    # -- the model ---------------------------------------------------------

    def _load(self):
        """Import graspnet-baseline and its weights. Heavy, and only here."""
        for folder in ('', 'models', 'dataset', 'utils'):
            path = os.path.join(GRASPNET, folder) if folder else GRASPNET
            if path not in sys.path:
                sys.path.append(path)
        import torch
        from graspnet import GraspNet, pred_decode
        from collision_detector import ModelFreeCollisionDetector

        self.torch = torch
        self.pred_decode = pred_decode
        self.CollisionDetector = ModelFreeCollisionDetector
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        checkpoint = self.get_parameter('checkpoint').value
        if not os.path.exists(checkpoint):
            raise SystemExit(
                f'{checkpoint} is not there. Fetch it with '
                f'grasp/fetch_graspnet.sh -- it is not in the repository, '
                f'because the licence does not let us redistribute it.')
        # The architecture the released weights were trained with. These are
        # not adjustable: they have to match the checkpoint.
        net = GraspNet(input_feature_dim=0,
                       num_view=self.get_parameter('num_view').value,
                       num_angle=12, num_depth=4, cylinder_radius=0.05,
                       hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04],
                       is_training=False)
        net.to(self.device)
        state = torch.load(checkpoint, map_location=self.device,
                           weights_only=False)
        net.load_state_dict(state['model_state_dict'])
        net.eval()
        self.net = net
        self.get_logger().info(
            f'loaded {os.path.basename(checkpoint)} '
            f'(epoch {state.get("epoch", "?")}) onto {self.device}')

    def _maybe_unload(self):
        after = self.get_parameter('idle_unload_after').value
        if not self._on_gpu or after <= 0 or self.device.type != 'cuda':
            return
        if time.time() - self._idle_since < after:
            return
        self.net.to('cpu')
        self._on_gpu = False
        try:
            self.torch.cuda.empty_cache()
        except Exception:                                # noqa: BLE001
            pass
        self.get_logger().info('idle, so the weights are off the GPU')

    def _ensure_loaded(self):
        if self._on_gpu:
            return
        self.net.to(self.device)
        self._on_gpu = True

    # -- inbound -----------------------------------------------------------

    def _on_color(self, msg):
        with self.lock:
            self._color = msg

    def _on_depth(self, msg):
        with self.lock:
            self._depth = msg

    def _on_info(self, msg):
        with self.lock:
            self._info = msg

    def _on_request(self, msg):
        try:
            payload = json.loads(msg.data)
        except ValueError:
            self.get_logger().warn(f'unreadable request: {msg.data[:80]!r}')
            return
        point = payload.get('point')
        if not point or len(point) != 3:
            self.get_logger().warn('a request needs a "point" in world metres')
            return
        with self.lock:
            self._pending = {'point': list(point),
                             'prompt': payload.get('prompt', '')}

    def _on_detections(self, msg):
        """The detector's own output is a request: it named an object and
        said where it is, which is everything needed to ask for grasps."""
        try:
            payload = json.loads(msg.data)
        except ValueError:
            return
        found = payload.get('detections') or []
        if not found:
            return
        with self.lock:
            if self._pending is None:
                self._pending = {'point': list(found[0]['point']),
                                 'prompt': payload.get('prompt', '')}

    # -- the work ----------------------------------------------------------

    def _loop(self):
        while not self._stop.is_set():
            with self.lock:
                request = self._pending
                self._pending = None
            if request is None:
                self._maybe_unload()
                self._stop.wait(0.2)
                continue
            self._ensure_loaded()
            started = time.time()
            try:
                self._serve(request)
            except Exception as exc:                     # noqa: BLE001
                self.get_logger().error(f'grasp inference failed: {exc}')
            self._idle_since = time.time()
            self.get_logger().info(
                f'grasps for {request["prompt"] or "the last detection"} in '
                f'{time.time() - started:.1f} s')

    def _serve(self, request):
        with self.lock:
            color, depth, info = self._color, self._depth, self._info
        if depth is None or info is None:
            self.get_logger().warn(
                'no depth or camera_info yet, so there is no cloud to grasp '
                'from', throttle_duration_sec=10.0)
            return

        cloud, colours = self._cloud(depth, color, info)
        if cloud is None or len(cloud) < 100:
            self.get_logger().warn('too few depth points to work from')
            return

        grasps = self._infer(cloud, colours)
        if grasps is None or not len(grasps):
            self._publish([], request, depth.header.frame_id)
            return

        try:
            rot, trans = self._camera_to_target(depth.header.frame_id,
                                                depth.header.stamp)
        except Exception as exc:                         # noqa: BLE001
            self.get_logger().warn(
                f'no transform {depth.header.frame_id} -> '
                f'{self.target_frame}: {exc}')
            return

        wanted = np.array(request['point'], dtype=np.float64)
        # 0 disables it: every grasp the model returns is considered,
        # wherever it is. The radius exists because the model proposes for
        # the whole scene and the cycle asked about one object, so without
        # it a grasp meant for the next thing along gets flown at this one.
        radius = self.get_parameter('object_radius').value
        floor = self.get_parameter('min_score').value
        widest = (self.get_parameter('gripper_open').value
                  - self.get_parameter('width_margin').value)
        margin = self.get_parameter('width_margin').value
        span_check = self.get_parameter('span_check').value
        ceiling = self.get_parameter('span_ceiling').value
        keep, too_wide, too_far, too_weak = [], 0, 0, 0
        rescued, clamped, near_widths = 0, 0, []
        for row in grasps:
            entry = self._to_world(row, rot, trans)
            if entry['score'] < floor:
                too_weak += 1
                continue
            if (radius > 0.0
                    and np.linalg.norm(
                        np.array(entry['position']) - wanted) > radius):
                too_far += 1
                continue
            near_widths.append(entry['width'])
            if entry['width'] > widest:
                if span_check:
                    # The model wants more opening than there is. Before
                    # refusing, ask the cloud how much object is actually
                    # between the jaws here -- that is the number the
                    # hardware has to satisfy, and it is often smaller.
                    # Measured first, then refused: never the other way
                    # round, which is what the ordering here is protecting.
                    span = self._span_between_jaws(row, cloud)
                    if span is None or span + margin > widest:
                        too_wide += 1
                        continue
                    entry['model_width'] = entry['width']
                    entry['width'] = round(span + margin, 4)
                    entry['width_source'] = 'measured'
                    rescued += 1
                else:
                    # Treated as an upper bound, not a requirement. Past
                    # the ceiling it is not this object at all and is still
                    # refused.
                    if entry['width'] > ceiling:
                        too_wide += 1
                        continue
                    entry['model_width'] = entry['width']
                    entry['width'] = round(widest, 4)
                    entry['width_source'] = 'clamped'
                    clamped += 1
            keep.append(entry)
        if clamped:
            self.get_logger().info(
                f'{clamped} grasp(s) wider than the {widest * 1000:.0f} mm '
                f'the jaws span were kept with the opening clamped to the '
                f'span -- span_check is off, so the width the model asked '
                f'for is an upper bound rather than a requirement')
        if rescued:
            self.get_logger().info(
                f'{rescued} grasp(s) the model called too wide are makeable: '
                f'the cloud says the object between the jaws is narrower '
                f'than the opening it asked for')
        if not keep:
            # "The model found grasps and this gripper cannot make any of
            # them" is a different problem from "the model found nothing",
            # and from "they were all somewhere else". Say which.
            limit = ceiling if not span_check else widest
            spread = ('' if not near_widths else
                      f' Near-object widths ran '
                      f'{min(near_widths) * 1000:.0f}-'
                      f'{max(near_widths) * 1000:.0f} mm.')
            self.get_logger().warn(
                f'nothing usable from {len(grasps)} raw grasp(s): '
                f'{too_weak} below score {floor:.2f}, '
                f'{too_far} further than {radius * 1000:.0f} mm from the '
                f'object, {too_wide} wider than the {limit * 1000:.0f} mm '
                f'{"span_ceiling" if not span_check else "these jaws span"}'
                f'.{spread}')
        keep.sort(key=lambda g: -g['score'])
        keep = keep[:self.get_parameter('max_candidates').value]
        self._publish(keep, request, self.target_frame)

    def _cloud(self, depth_msg, color_msg, info):
        """An organised point cloud in the camera frame, near points only."""
        depth = decode_image(depth_msg).astype(np.float32) * self.depth_scale
        height, width = depth.shape[:2]
        fx, fy = info.k[0], info.k[4]
        cx, cy = info.k[2], info.k[5]
        xmap, ymap = np.meshgrid(np.arange(width), np.arange(height))
        points_z = depth
        points_x = (xmap - cx) * points_z / fx
        points_y = (ymap - cy) * points_z / fy
        cloud = np.stack([points_x, points_y, points_z], axis=-1)

        near = self.get_parameter('min_depth').value
        far = self.get_parameter('max_depth').value
        mask = (points_z > near) & (points_z < far)
        cloud = cloud[mask].astype(np.float32)

        colours = None
        if color_msg is not None:
            frame = decode_image(color_msg)
            if frame.shape[:2] == depth.shape[:2]:
                colours = (frame[mask][:, ::-1] / 255.0).astype(np.float32)
        if colours is None:
            colours = np.zeros_like(cloud)
        return cloud, colours

    def _infer(self, cloud, colours):
        """Run the network. Returns the raw (N, 17) grasp array, or None."""
        wanted = int(self.get_parameter('num_point').value)
        if len(cloud) >= wanted:
            index = np.random.choice(len(cloud), wanted, replace=False)
        else:
            index = np.concatenate([
                np.arange(len(cloud)),
                np.random.choice(len(cloud), wanted - len(cloud),
                                 replace=True)])
        sampled = cloud[index]

        end_points = {
            'point_clouds': self.torch.from_numpy(
                sampled[np.newaxis].astype(np.float32)).to(self.device),
            'cloud_colors': colours[index],
        }
        with self.torch.no_grad():
            end_points = self.net(end_points)
            preds = self.pred_decode(end_points)
        grasps = preds[0].detach().cpu().numpy()

        threshold = self.get_parameter('collision_thresh').value
        if threshold > 0 and len(grasps):
            grasps = self._without_collisions(grasps, cloud, threshold)
        return grasps

    def _without_collisions(self, grasps, cloud, threshold):
        """Drop grasps whose gripper would be inside the cloud.

        Uses graspnet-baseline's own model-free detector, which wants an
        object with .translations/.rotation_matrices/.widths/.heights/.depths
        -- the raw array has all of that in known columns, so a small adapter
        avoids pulling in graspnetAPI for one class.
        """
        class _Group:
            translations = grasps[:, 13:16]
            rotation_matrices = grasps[:, 4:13].reshape(-1, 3, 3)
            heights = grasps[:, 2]
            depths = grasps[:, 3]
            widths = grasps[:, 1]

        detector = self.CollisionDetector(
            cloud, voxel_size=self.get_parameter('voxel_size').value)
        collided = detector.detect(_Group(), approach_dist=0.05,
                                   collision_thresh=threshold)
        return grasps[~collided]

    def _span_between_jaws(self, row, cloud):
        """How wide the object actually is where this grasp closes, metres.

        None when there is nothing between the jaws to measure.

        Why measure at all when the model reports a width: because that
        width is what GraspNet *recommends the gripper open to*, learned and
        binned over a 100 mm gripper's range. This one spans 44 mm, so the
        recommendation is coarse exactly where it has to be fine, and a
        candidate whose real material is 30 mm across can come back asking
        for 50. Measured, run 1789014831: every near-object candidate for a
        roll of tape was refused on the predicted width and the model has
        never contributed a pick on this robot.

        The cloud and the grasp are both in the camera frame, so this is
        done there: rotate the points into the grasp's own frame, keep the
        ones lying in the space the fingers will sweep through, and take
        their extent along the closing direction.

        The volume is graspnet-baseline's own, not a guess -- the same
        masks its collision detector builds (utils/collision_detector.py):
        the fingers occupy x in (depth - finger_length, depth), the object
        sits between them at |y| < width/2, and the gripper's thickness is
        |z| < height/2. FINGER_LENGTH is theirs too.
        """
        width, height, depth = row[1], row[2], row[3]
        r_cam = row[4:13].reshape(3, 3)
        t_cam = row[13:16]
        local = (cloud - t_cam) @ r_cam            # camera -> grasp frame
        between = (
            (local[:, 0] > depth - FINGER_LENGTH) & (local[:, 0] < depth)
            & (np.abs(local[:, 1]) < width / 2.0)
            & (np.abs(local[:, 2]) < max(height, 1e-3) / 2.0))
        if between.sum() < MIN_SPAN_POINTS:
            return None
        across = local[between, 1]
        return float(across.max() - across.min())

    def _to_world(self, row, rot, trans):
        """One raw grasp row, in the target frame and our tool's convention.

        The array's columns are
        [score, width, height, depth, R (9, row-major), translation (3), id].

        GraspNet's grasp frame, read off its own collision detector rather
        than assumed: local +x is the approach direction and the fingertips
        reach x = depth; local y is the closing direction, fingers at
        +/- width/2; local z is the gripper's thickness. So the point the
        jaws close around is translation + depth * R[:, 0].

        Our tool frame is different: openarm_<arm>_hand_tcp approaches along
        its own +Z. Mapping tool_z <- g_x and tool_y <- g_y (the closing
        direction, which is what a grasp is *about*) forces
        tool_x = tool_y x tool_z = -g_z, and that is the whole rotation.
        """
        score, width, depth = row[0], row[1], row[3]
        r_cam = row[4:13].reshape(3, 3)
        t_cam = row[13:16]

        centre_cam = t_cam + depth * r_cam[:, 0]
        tool_cam = np.stack([-r_cam[:, 2], r_cam[:, 1], r_cam[:, 0]], axis=1)

        centre = rot @ centre_cam + trans
        tool = rot @ tool_cam
        quat = rot_to_quat(tool)
        approach = tool[:, 2]
        return {
            'position': [round(float(v), 5) for v in centre],
            'quat': [round(float(v), 6) for v in quat],
            'width': round(float(width), 4),
            'score': round(float(score), 4),
            'depth': round(float(depth), 4),
            # How far off straight down the approach is, degrees. The
            # orchestrator's tilt limit is expressed the same way, so this is
            # the number that says whether a candidate is even a candidate.
            'tilt_deg': round(float(np.degrees(
                np.arccos(np.clip(-approach[2], -1.0, 1.0)))), 1),
        }

    def _camera_to_target(self, source_frame, stamp):
        tf = self.tf_buffer.lookup_transform(
            self.target_frame, source_frame, stamp,
            timeout=rclpy.duration.Duration(seconds=0.3))
        t = tf.transform.translation
        q = tf.transform.rotation
        return quat_to_rot(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z])

    def _publish(self, grasps, request, frame_id):
        payload = {
            'stamp': self.get_clock().now().nanoseconds * 1e-9,
            'frame_id': frame_id,
            'prompt': request.get('prompt', ''),
            'about': request['point'],
            'grasps': grasps,
        }
        self.candidates_pub.publish(String(data=json.dumps(payload)))
        if grasps:
            best = grasps[0]
            self.get_logger().info(
                f'{len(grasps)} grasp(s); best score {best["score"]:.2f} at '
                f'{best["position"]}, {best["width"] * 1000:.0f} mm opening, '
                f'{best["tilt_deg"]:.0f} deg off vertical')
        else:
            self.get_logger().warn(
                f'no grasp within '
                f'{self.get_parameter("object_radius").value * 1000:.0f} mm '
                f'of {request["point"]} survived scoring and collision '
                f'checking')

    def destroy_node(self):
        self._stop.set()
        self._worker.join(timeout=5.0)
        super().destroy_node()


def main():
    rclpy.init()
    node = GraspServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
