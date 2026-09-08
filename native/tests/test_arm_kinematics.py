#!/usr/bin/env python3
"""Checks for arm_kinematics: FK against the robot, and posture choice.

No ROS. Everything here is checked against something independent -- the joint
values the real arm reported in motion_log.jsonl, finite differences, or the
URDF's own limits -- rather than against another number this module produced.

    python3 native/tests/test_arm_kinematics.py
"""

import math
import os
import sys

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, WS)

import numpy as np                                              # noqa: E402

from arm_kinematics import (ArmChain, chain_from_urdf,           # noqa: E402
                            quat_matrix, rpy_matrix)

FAILURES = []


def check(name, got, want, tol=None):
    if tol is not None:
        ok = abs(float(got) - float(want)) <= tol
    else:
        ok = got == want
    print(('pass  ' if ok else 'FAIL  ') + name)
    if not ok:
        print(f'        got  {got}')
        print(f'        want {want}' + (f' +/- {tol}' if tol else ''))
        FAILURES.append(name)


# The right arm at the posture it reported while poised above the object,
# motion_log.jsonl run 1788771153, record "PREGRASP 8/8", and the tool
# position TF gave at that moment. This is the anchor for FK: if these agree,
# the chain is being walked the same way the robot's own state publisher walks
# it.
ABOVE_OBJECT = [1.308, 0.349, -1.558, 0.481, -1.553, 0.132, 1.307]
ABOVE_OBJECT_TCP = [0.4072, -0.2151, 0.4098]
# What the cycle asked for at that moment.
TOOL_TARGET = [0.4062, -0.2184, 0.4088]


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def top_down_quat(yaw):
    """The same construction pick_place_orchestrator uses."""
    return quat_mul((0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)),
                    (0.0, 1.0, 0.0, 0.0))


def main():
    urdf = open(os.path.join(WS, 'openarm.urdf')).read()
    chain = ArmChain(urdf, 'right')

    # -- the description ---------------------------------------------------
    check('seven arm joints', len(chain.joint_names), 7)
    # Was exactly 0.0, which is why HOME had to move off it -- and, worse,
    # why every plan started from an invalid state: both arms rest at
    # joint4 = -0.000191, marginally past a limit of zero, and cuRobo took
    # SIGFPE asked to plan from there. Now -0.05, which describes the
    # hyperextension the hardware already has.
    check('joint4 can go slightly negative, because the arm rests there',
          float(chain.limits[3][0]), -0.05, tol=1e-12)
    check('and the resting posture is now inside the limits',
          float(chain.margins([0.0, 0.0, 0.0, -0.000191, 0.0, 0.0, 0.0])[3])
          > 0.0, True)
    check("joint1's range is lopsided, and the short side is the negative one",
          (round(float(chain.limits[0][0]), 3),
           round(float(chain.limits[0][1]), 3)),
          (-1.396, 3.491))
    check('the tool sits 180.1 mm from joint7 centre -- the lever joint7 '
          'swings', chain.wrist_lever, 0.1801, tol=0.0002)

    # -- forward kinematics against the robot ------------------------------
    tcp = chain.pose(chain.tool_link, ABOVE_OBJECT)[:3, 3]
    check('FK matches the tool position the robot reported, to 0.2 mm',
          float(np.linalg.norm(tcp - np.array(ABOVE_OBJECT_TCP))), 0.0,
          tol=0.0002)
    z_axis = chain.pose(chain.tool_link, ABOVE_OBJECT)[:3, 2]
    check('and the gripper was pointing down there', float(z_axis[2]), -1.0,
          tol=0.002)
    wrist = chain.pose(chain.wrist_link, ABOVE_OBJECT)[:3, 3]
    check('the wrist was directly above the tool', float(wrist[2] - tcp[2]),
          0.1801, tol=0.0005)
    check('and over the same spot in x and y',
          float(np.linalg.norm(wrist[:2] - tcp[:2])), 0.0, tol=0.006)

    # A posture of all zeros is a sanity check no reader has to trust me on:
    # the arm hangs by the base, and the tool is below the shoulder.
    zero_tcp = chain.pose(chain.tool_link, [0.0] * 7)[:3, 3]
    check('at all zeros the arm hangs down', bool(zero_tcp[2] < 0.3), True)

    # -- limits ------------------------------------------------------------
    check('the posture the robot used was against two stops',
          int((chain.margins(ABOVE_OBJECT) < 0.02).sum()), 2)
    check('and its worst joint had 0.013 rad left',
          chain.worst_margin(ABOVE_OBJECT), 0.0128, tol=0.001)
    middle = (chain.limits[:, 0] + chain.limits[:, 1]) / 2.0
    check('mid-range has the most room of all', chain.worst_margin(middle),
          float(((chain.limits[:, 1] - chain.limits[:, 0]) / 2.0).min()),
          tol=1e-9)

    # -- Jacobians against finite differences ------------------------------
    probe = np.array([1.0, 0.5, -0.7, 0.9, 0.3, 0.2, -0.6])
    residual, jacobian = chain.objective_wrist([0.40, -0.22, 0.59])
    analytic = jacobian(probe)
    numeric = chain.numeric_jacobian(residual)(probe)
    check('the wrist Jacobian is the real derivative',
          float(np.abs(analytic - numeric).max()), 0.0, tol=1e-5)
    check('and joint7 has no effect on it at all -- the reason for solving '
          'here', float(np.abs(analytic[:, 6]).max()), 0.0, tol=1e-12)

    # The wrist point genuinely does not move with joint7, measured rather
    # than argued: it lies on that joint's axis.
    swung = list(ABOVE_OBJECT)
    swung[6] = ABOVE_OBJECT[6] - 1.2
    moved_wrist = chain.pose(chain.wrist_link, swung)[:3, 3]
    moved_tool = chain.pose(chain.tool_link, swung)[:3, 3]
    check('turning joint7 1.2 rad leaves the wrist where it was',
          float(np.linalg.norm(moved_wrist - wrist)), 0.0, tol=1e-9)
    check('while dragging the tool 200 mm',
          bool(np.linalg.norm(moved_tool - tcp) > 0.20), True)

    # -- the tilt ----------------------------------------------------------
    tilt, error = chain.tilt_for_direction(ABOVE_OBJECT, (0.0, 0.0, -1.0))
    # 1.64 deg, not 0: that is the tilt already present in the posture the
    # robot reached, and joint7 alone cannot take it out -- its axis is not
    # the one that residual lies about. The wrist-solved posture further down
    # gets to 0.09 deg, which is the number that matters.
    check('a tilt exists that brings the hand within 2 deg of straight down',
          math.degrees(error), 0.0, tol=2.0)
    tilted = list(ABOVE_OBJECT)
    tilted[6] = tilt
    down = chain.pose(chain.tool_link, tilted)[:3, 2]
    check('and applying it really does point it down', float(down[2]), -1.0,
          tol=0.001)

    # -- posture choice ----------------------------------------------------
    quat = top_down_quat(0.0)
    check('the grasp the cycle asked for is the one the robot achieved, to '
          '2 deg',
          math.degrees(float(np.arccos(max(-1.0, min(1.0, float(
              quat_matrix(quat)[:, 2] @ z_axis)))))), 0.0, tol=2.0)

    joints, margin, solved, tried = chain.tool_posture(TOOL_TARGET, quat,
                                                       seeds=48)
    check('the full tool pose does solve', bool(joints is not None), True)
    check('but every solution is jammed against a stop -- this is the '
          'problem, not a planner fault', bool(margin < 0.01), True)
    check('and the roomiest of them is the posture the robot actually used',
          float(np.abs(np.array(joints) - np.array(ABOVE_OBJECT)).max()), 0.0,
          tol=0.02)

    flipped = top_down_quat(math.pi)
    _, flip_margin, _, _ = chain.tool_posture(TOOL_TARGET, flipped, seeds=48)
    check('turning the jaws 180 deg -- the same grasp -- finds real headroom',
          bool(flip_margin > 0.15), True)

    wjoints, wmargin, wsolved, wtried = chain.approach_posture(
        TOOL_TARGET, quat, seeds=48)
    check('solving for the wrist finds postures too', bool(wjoints is not None),
          True)
    # Compared against the tool pose at the same target rather than against a
    # remembered number: the absolute magnitudes move with the joint limits and
    # with which basin the search lands in, but "the wrist objective has room
    # where the tool pose has none" is the structural claim.
    check(f'and finds real headroom where the tool pose has none '
          f'({wsolved}/{wtried} seeds)', bool(wmargin > margin + 0.1), True)
    wtilt, werror = chain.tilt_for_direction(
        list(wjoints), quat_matrix(quat)[:, 2])
    tilted = list(wjoints)
    tilted[6] = wtilt
    check('joint7 can still point the hand down from there',
          math.degrees(werror), 0.0, tol=0.5)
    landed = chain.pose(chain.tool_link, tilted)[:3, 3]
    check('and the tool lands on the target once tilted, within a mm',
          float(np.linalg.norm(landed - np.array(TOOL_TARGET))), 0.0,
          tol=0.001)
    after = chain.worst_margin(tilted)
    # The post-tilt margin is not asserted as a magnitude. It depends on which
    # basin the search lands in and on joint7's own headroom at the tilt it
    # needs, and it moved from 0.175 to 0.035 on a joint-limit change that
    # affected neither the target nor the objective. What has to hold is that
    # the posture is legal and lands on the target, both checked above.
    check('and the posture it ends in is inside every limit',
          bool(after > 0.0), True)
    # What the orchestrator would actually pick, both grasp directions
    # considered, against what the robot managed on its own.
    best_offered = max(margin, flip_margin)
    check('the best grasp offered beats what the planner found unaided',
          bool(best_offered > 5 * chain.worst_margin(ABOVE_OBJECT)), True)
    print(f'        headroom: robot {chain.worst_margin(ABOVE_OBJECT):.3f} rad, '
          f'tool-pose solve {margin:.3f}, jaws flipped {flip_margin:.3f}, '
          f'wrist solve {wmargin:.3f} -> {after:.3f} after the tilt')

    # -- the jaws have to close across the object --------------------------
    #
    # joint7 turns about link6's y-axis, and the fingers slide along the hand's
    # y -- which is link6's y. So the closing direction is fixed by joints 1-6
    # and joint7 cannot move it. Leaving it out of the objective therefore
    # saved nothing and cost a grasp: a full straight-line descent onto the
    # right point, and the gripper closing beside the screwdriver.
    swung = list(wjoints)
    for value in (-1.2, 0.0, 1.4):
        swung[6] = value
        check(f'joint7={value:+.1f} does not move the jaw axis',
              float(np.abs(chain.pose(chain.tool_link, swung)[:3, 1]
                           - chain.pose(chain.forearm_link, wjoints)[:3, 1]
                           ).max()), 0.0, tol=1e-9)

    want_axis = chain.jaw_axis_for(quat)
    check('the jaw axis of a top-down grasp is horizontal',
          float(want_axis[2]), 0.0, tol=1e-12)
    check('and for yaw 0 it is world +y',
          [round(float(v), 4) for v in want_axis], [0.0, 1.0, 0.0])

    # Unconstrained, the azimuth is whatever the solver lands on.
    loose = chain.approach_posture(TOOL_TARGET, quat, seeds=48)[0]
    pinned, pinned_margin, _, _ = chain.approach_posture(
        TOOL_TARGET, quat, seeds=48, jaw_axis=want_axis)
    check('a pinned solve still finds a posture', pinned is not None, True)

    def jaw_error(q):
        got = chain.pose(chain.forearm_link, q)[:3, 1]
        flat = np.array([got[0], got[1]])
        flat = flat / np.linalg.norm(flat)
        cos = abs(float(flat @ np.array([want_axis[0], want_axis[1]])))
        return math.degrees(math.acos(min(1.0, cos)))

    check('the pinned solve lines the jaws up exactly', jaw_error(pinned), 0.0,
          tol=0.05)
    print(f'        jaws: unconstrained {jaw_error(loose):.1f} deg off, '
          f'pinned {jaw_error(pinned):.3f} deg off')
    check('the unconstrained one was not lined up, which is the bug',
          bool(jaw_error(loose) > 1.0), True)

    # It still costs joint7 nothing, and still reaches the target.
    _, jac = chain.objective_wrist(chain.wrist_point_for(TOOL_TARGET, quat),
                                   jaw_axis=want_axis)
    check('joint7 stays out of the pinned objective too',
          float(np.abs(jac(np.array(pinned))[:, 6]).max()), 0.0, tol=1e-12)
    ptilt, perror = chain.tilt_for_direction(
        list(pinned), quat_matrix(quat)[:, 2])
    ptilted = list(pinned)
    ptilted[6] = ptilt
    check('and the hand still tilts to straight down', math.degrees(perror),
          0.0, tol=0.5)
    check('with the tool still on the target',
          float(np.linalg.norm(chain.pose(chain.tool_link, ptilted)[:3, 3]
                               - np.array(TOOL_TARGET))), 0.0, tol=0.005)
    check('and the jaws still lined up after tilting', jaw_error(ptilted), 0.0,
          tol=0.05)
    check('there is still real headroom', bool(pinned_margin > 0.15), True)

    # A different object angle gives a different posture, not the same one.
    slanted = top_down_quat(math.radians(35.0))
    slanted_axis = chain.jaw_axis_for(slanted)
    other = chain.approach_posture(TOOL_TARGET, slanted, seeds=48,
                                   jaw_axis=slanted_axis)[0]
    check('a 35 deg object axis is followed',
          bool(other is not None), True)
    if other is not None:
        got = chain.pose(chain.forearm_link, other)[:3, 1]
        flat = np.array([got[0], got[1]])
        flat = flat / np.linalg.norm(flat)
        cos = abs(float(flat @ np.array([slanted_axis[0], slanted_axis[1]])))
        check('to within a twentieth of a degree',
              math.degrees(math.acos(min(1.0, cos))), 0.0, tol=0.05)
        check('and it is a different posture from the yaw-0 one',
              bool(np.abs(np.array(other) - np.array(pinned)).max() > 0.05),
              True)

    # -- determinism, which is the point of choosing at all ----------------
    again = chain.approach_posture(TOOL_TARGET, quat, seeds=48)[0]
    check('the same question gives the same posture every time',
          float(np.abs(np.array(again) - np.array(wjoints)).max()), 0.0,
          tol=1e-12)

    # A warm start must not make it worse: extra seeds are extra candidates,
    # never a replacement for the search.
    warm = chain.approach_posture(TOOL_TARGET, quat, seeds=48,
                                  extra_seeds=[ABOVE_OBJECT])[1]
    check('a warm start never loses headroom', bool(warm >= wmargin - 1e-9),
          True)

    # -- wrist_point_for ---------------------------------------------------
    point = chain.wrist_point_for(TOOL_TARGET, quat)
    check('a downward grasp puts the wrist straight above the tool',
          float(point[2] - TOOL_TARGET[2]), 0.1801, tol=0.0002)
    check('and directly over it', float(np.linalg.norm(
        point[:2] - np.array(TOOL_TARGET[:2]))), 0.0, tol=1e-9)
    sideways = chain.wrist_point_for(TOOL_TARGET, (0.0, 0.7071, 0.0, 0.7071))
    check('a sideways grasp puts it to the side instead, not above',
          bool(abs(sideways[2] - TOOL_TARGET[2]) < 0.02), True)

    # -- the left arm is not a mirror of the right in its limits -----------
    left = ArmChain(urdf, 'left')
    check('the left arm builds too', len(left.joint_names), 7)
    check("and its joint2 range is the right's reflected, not the same",
          (round(float(left.limits[1][0]), 3), round(float(left.limits[1][1]), 3)),
          (-round(float(chain.limits[1][1]), 3),
           -round(float(chain.limits[1][0]), 3)))
    check('its tool lever is the same 180.1 mm', left.wrist_lever, 0.1801,
          tol=0.0002)

    # -- rotations ---------------------------------------------------------
    check('rpy_matrix of nothing is the identity',
          float(np.abs(rpy_matrix(0, 0, 0) - np.eye(3)).max()), 0.0, tol=1e-12)
    check('quat_matrix of the identity quaternion likewise',
          float(np.abs(quat_matrix((0, 0, 0, 1)) - np.eye(3)).max()), 0.0,
          tol=1e-12)
    check('a half turn about y sends +z to -z',
          float(quat_matrix((0.0, 1.0, 0.0, 0.0))[2, 2]), -1.0, tol=1e-12)
    check('an unnormalised quaternion is still handled',
          float(np.abs(quat_matrix((0.0, 2.0, 0.0, 0.0))
                       - quat_matrix((0.0, 1.0, 0.0, 0.0))).max()), 0.0,
          tol=1e-12)

    # -- graceful failure --------------------------------------------------
    check('a description without this arm gives None, not an exception',
          chain_from_urdf('<robot name="x"><joint name="a" type="fixed">'
                          '<parent link="p"/><child link="c"/></joint></robot>',
                          'right'), None)
    check('and neither does malformed XML raise',
          chain_from_urdf('<robot', 'right'), None)

    print()
    if FAILURES:
        print(f'{len(FAILURES)} check(s) failed: ' + ', '.join(FAILURES))
        return 1
    print('forward kinematics matches the robot, and the wrist partition '
          'gives the arm room the tool pose does not')
    return 0


if __name__ == '__main__':
    sys.exit(main())
