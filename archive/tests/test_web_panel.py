#!/usr/bin/env python3
"""Checks for the web panel: the motor report, and the page it is drawn on.

Two halves, and neither needs the robot.

The first serves a fake /joint_states and asks pick_place_web what it makes
of it. That is the part worth testing, because "is this motor alive" is not
a flag anyone publishes -- openarm_hardware exports position, velocity and
effort per joint and nothing else, so a motor that has fallen off the CAN
bus keeps appearing on the topic carrying the values it had when it went.
The report infers it from that, and an inference deserves a test.

The second half is about the page. Every element the script reaches for by
id has to exist in the markup, and the script has to parse -- both of which
have broken here before: `const held` sat below its own first use, so every
status message threw out of the temporal dead zone and the panel froze
after its first paint with no error anywhere a user would look.

    source native/setup.bash && python3 native/tests/test_web_panel.py
"""

import json
import os
import re
import subprocess
import sys
import threading
import time

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, WS)

# Isolate from any live robot before rclpy creates a participant: this test
# publishes /joint_states, and on the default domain that would fight with
# the real arms' own publisher.
os.environ['ROS_DOMAIN_ID'] = os.environ.get(
    'WEB_TEST_DOMAIN', str(41 + os.getpid() % 20))
os.environ['ROS_LOCALHOST_ONLY'] = '1'

import rclpy                                                     # noqa: E402
from rclpy.executors import MultiThreadedExecutor                # noqa: E402
from rclpy.node import Node                                      # noqa: E402
from control_msgs.msg import (                                    # noqa: E402
    DynamicJointState, InterfaceValue, JointTrajectoryControllerState)
from sensor_msgs.msg import JointState                           # noqa: E402

import pick_place_web as panel                                   # noqa: E402

PAGE = os.path.join(WS, 'pick_place_web.html')

FAILURES = []


def check(label, got, want):
    ok = got == want
    print(('pass  ' if ok else 'FAIL  ') + label)
    if not ok:
        print(f'        got  {got}')
        print(f'        want {want}')
        FAILURES.append(label)


NAMES = ([f'openarm_left_joint{i}' for i in range(1, 8)]
         + ['openarm_left_finger_joint1']
         + [f'openarm_right_joint{i}' for i in range(1, 8)]
         + ['openarm_right_finger_joint1'])

# The one that is modelled as gone. It keeps its place in msg.name and its
# three numbers never change again, which is exactly what the real thing
# looks like: read() copies the last frame it received, for ever.
DEAD = 'openarm_left_joint3'

# And the one modelled as limp: reporting perfectly, numbers moving, no
# torque holding it. Measured on this robot, both arms parked: every joint
# sat within 0.00038 rad of its commanded position except two, at 0.00286
# and 0.00506 -- and one of those was showing a green light, because its
# readings were changing more than a driven joint's, not less.
ADRIFT = 'openarm_left_joint7'
ADRIFT_ERROR = 0.006


class FakeJoints(Node):
    """A robot at rest, as the real one reports it.

    Not perfectly still, which is the whole point. Measured in
    motion_log.jsonl with the arm holding a pose: joint1's effort read
    3.5473, 3.3099, 3.4418 and 3.9429 Nm on four consecutive stills, and
    the velocities sat at +/-0.01 rad/s rather than at zero. A direct-drive
    motor holding an arm up against gravity never repeats itself.
    """

    def __init__(self):
        super().__init__('fake_joint_states')
        self.pub = self.create_publisher(JointState, '/joint_states', 10)
        # Set by the checks to make the controllers' reference move, which
        # is what "the arm is flying a trajectory" looks like from here.
        self.moving = False
        # What each motor reports about itself, keyed by joint. Empty
        # means the driver does not export a status interface at all,
        # which is the pre-rebuild stack and has to stay supported.
        self.reports = {}
        self.dynamic = self.create_publisher(
            DynamicJointState, '/dynamic_joint_states', 10)
        self.control = {
            arm: self.create_publisher(
                JointTrajectoryControllerState,
                f'/{arm}_joint_trajectory_controller/controller_state', 10)
            for arm in ('left', 'right')}
        self.n = 0
        self.create_timer(0.05, self.tick)

    def tick(self):
        self.n += 1
        msg = JointState()
        msg.name = list(NAMES)
        msg.position = [0.1 if name == DEAD else 0.1 + self.n * 1e-4
                        for name in NAMES]
        msg.velocity = [0.0 if name == DEAD else 0.01 * ((self.n % 3) - 1)
                        for name in NAMES]
        msg.effort = [1.0 if name == DEAD else 1.0 + (self.n % 7) * 0.01
                      for name in NAMES]
        self.pub.publish(msg)
        # And what each controller says about the joints it drives. The
        # reference never moves -- the arms are parked -- so every error
        # here is a joint failing to hold, not one lagging a trajectory.
        for arm in ('left', 'right'):
            state = JointTrajectoryControllerState()
            state.joint_names = [f'openarm_{arm}_joint{i}'
                                 for i in range(1, 8)]
            goal = 0.1 + (self.n * 1e-3 if self.moving else 0.0)
            state.reference.positions = [goal] * 7
            state.feedback.positions = [goal] * 7
            state.error.positions = [
                ADRIFT_ERROR if name == ADRIFT else 0.0001
                for name in state.joint_names]
            self.control[arm].publish(state)
        if self.reports:
            dyn = DynamicJointState()
            for name, code in self.reports.items():
                dyn.joint_names.append(name)
                value = InterfaceValue()
                value.interface_names = ['position', 'status']
                value.values = [0.1, float(code)]
                dyn.interface_values.append(value)
            self.dynamic.publish(dyn)


def check_motor_report():
    rclpy.init()
    bridge = panel.Bridge()
    fake = FakeJoints()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(bridge)
    executor.add_node(fake)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        # Long enough for the controllers' reference to count as settled,
        # or every joint is judged mid-move and none of them is judged at
        # all.
        time.sleep(panel.MOTOR_HOLD_SETTLE + 1.5)
        report = bridge.motor_report()
        rows = {row['name']: row for row in report['joints']}
        check('every motor on both arms is reported', report['total'], 16)

        # The failure a freshness check cannot see, and the reason the
        # controller is asked as well: this motor reports beautifully. Its
        # numbers change on every frame -- more than a driven joint's,
        # because gravity is moving it -- and it is nowhere near where it
        # was told to be.
        check('a joint reporting fine but not holding its position is adrift',
              rows[ADRIFT]['state'], 'adrift')
        check('and the panel is told how far off it is',
              rows[ADRIFT]['holding'], False)
        check('in radians, from the controller itself',
              round(rows[ADRIFT]['error'], 5), ADRIFT_ERROR)
        check('while a joint inside the threshold is holding',
              rows['openarm_right_joint4']['holding'], True)
        check('which is the worst thing wrong so far', report['worst'],
              'adrift')
        check('the panel knows a controller is publishing at all',
              report['tracked'], True)
        # The gripper is not on the trajectory controller, so there is
        # nothing to judge it against and it is not judged -- and with no
        # status from the motor either, "unknown" is the whole of what can
        # honestly be said about it.
        check('a joint no controller reports on is left unjudged',
              rows['openarm_left_finger_joint1']['holding'], None)
        check('and is not called healthy on the strength of nothing',
              rows['openarm_left_finger_joint1']['state'], 'unknown')
        # 16 motors: 13 arm joints answering and holding, one adrift, and
        # the two grippers that nothing can vouch for.
        check('all the rest are live', report['live'], 13)

        # And the verdict survives the arm starting to move. It cannot be
        # re-measured mid-move -- the error is then the arm lagging its own
        # trajectory -- but a motor with no torque does not heal because a
        # cycle started, and a light that goes green the moment the robot
        # is doing something is a light that is green whenever anyone is
        # watching it work.
        fake.moving = True
        time.sleep(1.0)
        moving = {row['name']: row
                  for row in bridge.motor_report()['joints']}
        check('a joint that failed while parked stays failed once the arm '
              'moves', moving[ADRIFT]['state'], 'adrift')
        check('and says when it was actually measured',
              moving[ADRIFT]['judged'] is not None, True)
        check('while a joint that was holding is simply not judged mid-move',
              moving['openarm_right_joint4']['holding'], None)
        fake.moving = False
        time.sleep(panel.MOTOR_HOLD_SETTLE + 1.0)

        # -- what the motor says about itself ------------------------------
        #
        # Everything above this is inference from an encoder, and an
        # encoder reads the same whether or not there is torque behind it.
        # A joint carrying almost no load -- a wrist roll near neutral --
        # holds its place on friction alone, so the tracking check passes
        # it and the panel showed green for a motor with no torque at all.
        # That is the case this answers: the Damiao motors report their own
        # status in every feedback frame, openarm_can was discarding the
        # byte, and openarm_hardware now exports it per joint.
        before = bridge.motor_report()
        check('with nothing reporting, the panel does not claim to know',
              before['reported'], False)
        idle = {row['name']: row for row in before['joints']}
        check('and a joint it cannot judge is not called healthy',
              idle['openarm_right_finger_joint1']['state'], 'unknown')

        fake.reports = {name: 1 for name in NAMES}
        fake.reports[ADRIFT] = 0                      # reports itself off
        fake.reports['openarm_right_joint5'] = 14     # overloaded
        time.sleep(1.2)
        told = bridge.motor_report()
        rows = {row['name']: row for row in told['joints']}
        check('a reported status is read straight off the motor',
              told['reported'], True)
        check('a motor that says it is not enabled is not green',
              rows[ADRIFT]['state'], 'off')
        check('and its own word outranks the tracking inference',
              rows[ADRIFT]['says'], 'not enabled')
        check('a fault is named rather than lumped in with the rest',
              rows['openarm_right_joint5']['says'], 'overloaded')
        check('and shows as a fault', rows['openarm_right_joint5']['state'],
              'fault')
        check('a motor reporting enabled and tracking is simply live',
              rows['openarm_right_joint2']['state'], 'live')
        check('with the raw code kept, so an unrecognised one is visible',
              rows['openarm_right_joint2']['reported'], 1)
        check('and the fault is what the summary leads with',
              told['worst'], 'fault')

        # A stale report is not a current one.
        fake.reports = {}
        time.sleep(panel.MOTOR_STATUS_AGE + 1.0)
        stale = bridge.motor_report()
        check('a status that stopped arriving is not kept as fact',
              {row['name']: row for row in stale['joints']}[
                  'openarm_right_joint5']['says'], None)
        check('the gripper is labelled as jaws rather than joint8',
              rows['openarm_left_finger_joint1']['label'], 'jaws')
        check('and an arm joint by its number',
              rows['openarm_right_joint4']['label'], 'j4')
        check('each row carries what the motor last said',
              round(rows['openarm_right_joint1']['effort'], 2) >= 1.0, True)

        # Long enough for the frozen one to cross the line, and not a
        # millisecond of it arbitrary: it is the threshold the report uses.
        time.sleep(panel.MOTOR_QUIET_SECONDS + 0.6)
        report = bridge.motor_report()
        stuck = sorted(row['name'] for row in report['joints']
                       if row['state'] == 'quiet')
        check('a motor still on the topic but saying the same thing is amber',
              stuck, [DEAD])
        check('and it is one of several the panel now has something to '
              'say about', report['live'], 12)
        check('with the frozen one outranking the adrift one, because a '
              'motor that has stopped reporting is the worse of the two',
              report['worst'], 'quiet')
        check('with how long it has been saying it',
              report['joints'][2]['quiet'] > panel.MOTOR_QUIET_SECONDS, True)
        check('no controller manager here, so nothing is claimed about power',
              report['components'], [])

        # The topic stopping is a different failure and reads differently:
        # it is not one motor, it is the bus or the bringup.
        fake.destroy_node()
        executor.remove_node(fake)
        time.sleep(panel.MOTOR_TOPIC_SECONDS + 0.6)
        report = bridge.motor_report()
        check('a stopped /joint_states makes every motor silent',
              report['live'], 0)
        check('and says silent rather than amber', report['worst'], 'silent')
        check('reporting the age of the last message, not a guess',
              report['topic_age'] > panel.MOTOR_TOPIC_SECONDS, True)

        # It goes down a websocket, so it has to survive json.
        json.dumps(report)
        check('the snapshot a freshly opened page gets carries it too',
              'motors' in bridge.snapshot(), True)
    finally:
        executor.shutdown()
        bridge.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def check_page():
    """The page's script against the page's markup."""
    with open(PAGE, encoding='utf-8') as handle:
        html = handle.read()

    ids = set(re.findall(r'\bid="([^"]+)"', html))
    wanted = set(re.findall(r"\$\('([^']+)'\)", html))
    missing = sorted(wanted - ids)
    check('every element the script reaches for by id exists', missing, [])

    # The classes the script sets by name have to appear in the style
    # sheet, or a state is set and nothing on screen shows it. Checked
    # against the <style> block alone -- "live" appears in half the prose
    # on the page, and a substring match against the whole file would pass
    # whatever it was given.
    style = html[html.index('<style>'):html.index('</style>')]
    unstyled = [name for name in
                ('pill', 'live', 'bad', 'warn', 'ok', 'primary', 'grow',
                 'motor', 'dead', 'off', 'silent', 'node', 'done', 'fixed',
                 'dot', 'led', 'hint', 'card', 'head', 'flow', 'divider')
                if not re.search(r'\.' + name + r'\b', style)]
    check('every class the script sets is styled', unstyled, [])

    # hidden has to hide. The UA sheet's [hidden] rule is the weakest thing
    # in the cascade, and `#camera { display: block }` and `.row { display:
    # flex }` both beat it -- which put a broken-image icon above the "no
    # frame yet" box and showed the per-hand Place pair beside the single
    # Place button, with element.hidden set on both.
    check('[hidden] is enforced against the page\'s own display rules',
          '[hidden] { display: none !important; }' in style, True)

    # And the script has to parse. node is not a hard dependency of this
    # workspace, so a machine without it skips rather than fails.
    script = html[html.index("<script>\n'use strict';") + len('<script>\n'):
                  html.rindex('</script>')]
    path = os.path.join(WS, 'native', 'tests',
                        f'.web_panel.{os.getpid()}.js')
    try:
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write(script)
        try:
            done = subprocess.run(['node', '--check', path],
                                  capture_output=True, text=True)
        except FileNotFoundError:
            print('skip: node is not installed, so the script was not parsed')
            return
        check('the page script parses', (done.returncode, done.stderr.strip()),
              (0, ''))
    finally:
        if os.path.exists(path):
            os.remove(path)

    # The one that actually bit: a const read above its own declaration is
    # perfectly legal to parse and throws at run time, so `node --check`
    # passing says nothing about it. The first mention of `held` in
    # renderStatus has to be the one that declares it.
    render = html[html.index('function renderStatus('):
                  html.index('/* ------------------------------------------'
                             '------------------ flow chart */')]
    declared = render.index('const held =') + len('const ')
    first_use = min(m.start() for m in re.finditer(r'\bheld\b', render))
    check('renderStatus declares held before it reads it -- the temporal '
          'dead zone that froze the panel',
          first_use, declared)


def main():
    print('-- what the motor report makes of a robot at rest --')
    check_motor_report()
    print('\n-- the page --')
    check_page()
    print()
    if FAILURES:
        print(f'{len(FAILURES)} check(s) failed: {", ".join(FAILURES)}')
        return 1
    print('the panel reports what the motors are doing, and the page it is '
          'drawn on holds together')
    return 0


if __name__ == '__main__':
    sys.exit(main())
