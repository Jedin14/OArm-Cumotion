#!/usr/bin/env python3
"""Watch /grasp/candidates and say whether the grasps make sense.

The one thing that cannot be checked by reading code is whether the frame
mapping is right. graspnet-baseline reports grasps in its own convention --
local +x the approach, +y the closing direction -- and grasp_node maps that to
the tool's (+Z the approach). A sign error there produces confident,
well-scored grasps pointing sideways or up, and the first sign of it on real
hardware would be the wrist driving somewhere unexpected.

So: run this against a live camera with the arm simulated, put an object on
the table, and read the numbers.

    grasp/check_grasps.py                  # watch, forever
    grasp/check_grasps.py --once           # one set and exit
    grasp/check_grasps.py --ask 0.32 0.07 0.35   # request grasps at a point

What to look for, in order of how badly each one bites:

  approach   should be near 0 deg for a tabletop object -- straight down. A
             cluster near 180 means the approach axis is inverted; near 90
             means x and z are swapped.
  height     the grasp z should sit within a few cm of the detector's point.
             Far below the table means the mapping put the tool through it.
  width      should be a plausible opening for the object, and under the
             gripper's 44 mm to be usable at all.
  agreement  the spread between the top candidates. All pointing the same way
             is a good sign; scattered is either a hard object or a bug.
"""

import argparse
import json
import math
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String

GRIPPER_OPEN = 0.044          # what the hardware can actually span


def describe(payload, detection):
    """One set of candidates, judged rather than dumped."""
    grasps = payload.get('grasps') or []
    about = payload.get('about')
    lines = [
        f'--- {len(grasps)} grasp(s) for {payload.get("prompt") or "?"} '
        f'about {[round(v, 3) for v in about] if about else "?"} '
        f'in {payload.get("frame_id")}',
    ]
    if not grasps:
        lines.append('    nothing. Either the object is outside '
                     'object_radius of the detection, or every candidate '
                     'was below min_score or collided.')
        return '\n'.join(lines), False

    lines.append(f'    {"score":>6} {"tilt":>6} {"width":>7} {"position":>26}'
                 f'  {"approach (unit)":>22}')
    ok = True
    for grasp in grasps[:8]:
        x, y, z = grasp['position']
        tilt = grasp['tilt_deg']
        width = grasp['width']
        # The approach direction back out of the quaternion, as the tool's
        # own +Z, so this is checking the published orientation rather than
        # re-reporting tilt_deg from the same arithmetic.
        qx, qy, qz, qw = grasp['quat']
        ax = 2 * (qx * qz + qy * qw)
        ay = 2 * (qy * qz - qx * qw)
        az = 1 - 2 * (qx * qx + qy * qy)
        flag = ''
        if tilt > 60:
            flag += '  <- not pointing down'
            ok = False
        if width > GRIPPER_OPEN:
            flag += f'  <- wider than the {GRIPPER_OPEN * 1000:.0f} mm jaws'
            ok = False
        lines.append(
            f'    {grasp["score"]:6.3f} {tilt:5.0f}d {width * 1000:6.1f}mm '
            f'[{x:+.3f} {y:+.3f} {z:+.3f}]  '
            f'[{ax:+.2f} {ay:+.2f} {az:+.2f}]{flag}')

    tilts = [g['tilt_deg'] for g in grasps]
    lines.append(f'    tilt spread {min(tilts):.0f}..{max(tilts):.0f} deg')
    if about:
        drops = [g['position'][2] - about[2] for g in grasps]
        lines.append(
            f'    height vs the detected point: '
            f'{min(drops) * 1000:+.0f}..{max(drops) * 1000:+.0f} mm')
    if detection:
        gap = math.dist(grasps[0]['position'], detection)
        lines.append(f'    best candidate is {gap * 1000:.0f} mm from the '
                     f'detector\'s point')
    if min(tilts) > 60:
        lines.append('    EVERY candidate points sideways or up. If the '
                     'object is on a table this is the frame mapping, not '
                     'the model -- see _to_world in grasp_node.py.')
    return '\n'.join(lines), ok


class Watcher(Node):
    """Print each set of candidates as it arrives, with a verdict."""

    def __init__(self, args):
        super().__init__('check_grasps')
        self.args = args
        self.seen = 0
        self.detection = None
        latched = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/grasp/candidates',
                                 self._on_candidates, latched)
        self.create_subscription(String, '/vlm/detections',
                                 self._on_detections, 10)
        self.ask = self.create_publisher(String, '/grasp/request', 10)
        if args.ask:
            self.create_timer(1.0, self._request)

    def _on_detections(self, msg):
        try:
            found = (json.loads(msg.data).get('detections') or [])
        except ValueError:
            return
        if found:
            self.detection = found[0]['point']

    def _on_candidates(self, msg):
        try:
            payload = json.loads(msg.data)
        except ValueError:
            print('unreadable payload')
            return
        text, ok = describe(payload, self.detection)
        print(text, flush=True)
        self.seen += 1
        if self.args.once:
            raise SystemExit(0 if ok else 1)

    def _request(self):
        self.ask.publish(String(data=json.dumps(
            {'point': self.args.ask, 'prompt': 'manual request'})))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true',
                        help='exit after one set; non-zero if it looks wrong')
    parser.add_argument('--ask', nargs=3, type=float, metavar=('X', 'Y', 'Z'),
                        help='request grasps about this world point')
    args = parser.parse_args()

    rclpy.init()
    node = Watcher(args)
    print('watching /grasp/candidates ... (start a pick, or use --ask)',
          file=sys.stderr)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit) as stop:
        return getattr(stop, 'code', 0) or 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
