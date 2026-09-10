"""VLM-guided pick and place, launched on top of a running launch_everything.

Bring the robot up first (native/run_launch_everything.sh), then start this in a
second terminal:

    source native/setup.bash
    ros2 launch pick_place.launch.py prompt:="detect screwdriver"

It is deliberately separate from launch_everything.launch.py for two reasons:
PaliGemma takes tens of seconds and several GB of VRAM to load, which you do not
want on every robot bringup; and the two run in different Python stacks --
vlm_detector_node.py needs VLM/.venv (torch 2.14+cu130) while everything else
needs native/venv (torch 2.7+cu128, the ABI cuRobo's kernels are built against).
run_vlm_detector.sh is what keeps those apart, so the detector is started through
that wrapper rather than as a launch_ros Node.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration

WS = os.path.dirname(os.path.realpath(__file__))

# Orchestrator parameters that are worth exposing as launch arguments. Everything
# else is declared in pick_place_orchestrator.py and can be overridden with
# --ros-args -p on a manual run.
ORCHESTRATOR_ARGS = [
    ('arm', 'right',
     'Which arm to use. It must match the tool_frame the robot was launched '
     'with -- cuMotion takes Cartesian goals for one link only, and the '
     'orchestrator refuses to start on a mismatch.'),
    ('home_joint_positions', '[0.0, 0.0, 0.0, 0.20, 0.0, 0.0, 0.0]',
     'HOME, joint1..joint7 in radians: where the arm rests, observes from, '
     'and returns to, and the only pose the octomap is captured from. This is '
     'the "home" group state in the SRDF -- which folds the arm out of the '
     'camera view -- except for joint4. The SRDF asks that joint for 0.0, '
     'which is exactly its URDF lower limit, and the elbow stops 8.9 degrees '
     'short of it; 0.20 rad clears the floor so the posture can actually be '
     'held. Jog somewhere else and call /pick_place/capture_ready for its '
     'numbers.'),
    ('arm_selection', 'by_side',
     '"by_side" takes the half of the camera frame the object is in and gets '
     'on with it -- the camera half is right nearly every time, and asking '
     'the solvers instead cost 87 seconds between the first look and the '
     'second, measured, for the same answer. Nothing is given up: the '
     'pre-flight settles whether the pick is possible for the arm chosen, '
     'before anything moves, so a wrong guess costs a pre-flight rather than '
     'a motion. "by_reach" asks the solvers first; "fixed" always uses arm.'),
    ('arm_order', 'camera_half',
     'Order the arms are considered in: "camera_half" prefers the half the '
     'object is in and falls back to the other arm; "right_then_left" and '
     '"left_then_right" are fixed orders. The first arm that can reach the '
     'object gets it.'),
    ('arm_split_px', '0.0',
     'Nudge for the camera-half split, pixels added to the middle column. '
     'Positive widens the left arm half. Only needed if the camera is not '
     'centred on the robot.'),
    ('arm_split_y', '0.0',
     'Fallback split on world y, used only when the detector publishes no '
     'image_size. The camera half is what normally decides.'),
    ('states_file', 'auto',
     'YAML of poses recorded by record_states.py; "auto" means '
     'pick_place_states_<arm>.yaml. pre_pick_state is always needed; '
     'drop_state is needed when place_mode is "state".'),
    ('place_mode', 'state',
     '"state" releases at the recorded drop_state; "ready" releases at the '
     'observation pose; "position" moves over place_position and releases '
     'there.'),
    ('place_position', '[0.35, 0.30, 0.25]',
     'Where to drop the object, xyz in the world frame. Only used when '
     'place_mode is "position". Measure this in RViz.'),
    ('place_yaw', '0.0',
     'Tool yaw when releasing, radians. place_mode "position" only.'),
    ('approach_height', '0.05',
     'Pre-grasp height above the grasp, metres: the arm stops here, opens the '
     'gripper, then descends.'),
    ('transit_height', '0.15',
     'Height above the grasp at which the long free-space move ends, metres. '
     'Nothing plans a straight line, so a single move to a point just above '
     'the object can arrive from the side and push it away; below this height '
     'the tool descends the vertical line above the object instead. Set 0 to '
     'go straight to the pre-grasp.'),
    ('linear_descent', 'true',
     'Ask /compute_cartesian_path for a genuine straight line below '
     'transit_height. A goal pose says where to end up, not how to get there, '
     'and a 5 cm descent planned as a free trajectory can bow into the table.'),
    ('cartesian_step', '0.005',
     'Interpolation step for that line, metres. Smaller is straighter.'),
    ('retreat_height', '0.20',
     'How high the object is lifted before being carried anywhere, metres '
     'above the grasp. The carry to the drop pose is a free-space plan, and '
     'starting it 5 cm above the surface dragged the gripper across the table. '
     'Defaults to transit_height, so the object leaves at the altitude the '
     'approach arrived at. 0 means back to the pre-grasp only.'),
    ('approach_ignores_octomap', 'true',
     'Let the final descent and the retreat ignore the collision world when a '
     'checked straight line cannot be had. On a top-down grasp the target is '
     'itself in the octomap -- the gripper must enter the voxels of the object '
     'it is picking up -- so a checked descent onto an object never completes. '
     'Measured: the 20 cm drop to the pre-grasp solved 100% of its line, the '
     'last 5 cm solved 25%. The leg is short, straight and vertical between '
     'reach-checked ends with min_grasp_z as a hard floor; the alternative is '
     'a free-space plan for the same 5 cm, which drove the gripper into the '
     'table.'),
    ('stage_drop_through_pre_pick', 'true',
     'Carry the object out through the pre-pick pose instead of going '
     'straight from the lift to the drop. The lift ends low over the work '
     'surface and the drop pose is across it; a direct joint goal came back '
     'INVALID_MOTION_PLAN three times, which is what a path swinging through '
     'the octomap looks like. pre_pick is above the table by construction and '
     'is already the waypoint used on the way back.'),
    ('gripper_octomap_exemption', 'true',
     'Exempt only the gripper links from the octomap, rather than switching '
     'collision checking off for the whole arm. The descent needs an '
     'exemption for one narrow reason -- on a top-down grasp the object being '
     'picked up is itself in the map, so the fingers must enter its voxels -- '
     'and that says nothing about the forearm or the elbow, which with '
     'checking off entirely are free to sweep into the table. Applied as an '
     'allowed-collision-matrix entry for one leg and withdrawn afterwards.'),
    ('descend_ignores_octomap', 'true',
     'On the vertical column legs, do not ask for a collision-checked line at '
     'all -- go straight to the unchecked one. On a top-down grasp the target '
     'is itself in the octomap, so the checked line stalls about a centimetre '
     'above the object whatever the voxel size; asking first costs a planning '
     'round trip and can fly a partial checked line that leaves the tool '
     'somewhere the rest does not solve from. Per leg: only where '
     'approach_ignores_octomap already grants the exemption.'),
    ('cartesian_min_fraction', '0.98',
     'Refuse a partial Cartesian path rather than execute it -- a descent that '
     'stops short leaves the gripper closing on air.'),
    ('descend_step', '0.02',
     'Longest vertical hop below transit_height, metres. Waypoints are '
     'collinear above the object, so short hops cannot bow far off that line. '
     'Set 0 to disable stepping.'),
    ('seeded_descent', 'true',
     'Solve the descent with IK seeded from the posture above each waypoint '
     'and send joint goals. Seven joints for a six-DOF pose means a pose goal '
     'does not pick a posture, so the planner will reconfigure the whole arm '
     'to lower the tool a few centimetres. Set false for the old behaviour.'),
    ('max_joint_jump', '0.5',
     'Reject an IK solution that moves any joint further than this from its '
     'seed, radians -- that is a reconfiguration, not a descent.'),
    ('ik_timeout', '1.0', 'Timeout passed to /compute_ik, seconds.'),
    ('plan_attempts', '3',
     "Times to resend a goal that failed for a retryable reason. cuMotion's "
     'optimiser returns TRAJOPT_FAIL on roughly 14.5% of goals that plan fine '
     'on another try.'),
    ('pick_place_config', 'pick_place_config.json',
     'Where the editable sequence and the UI-settable parameters live. Read '
     'at bringup and written by the web UI, so a run starts where the last '
     'one left off. Empty uses the built-in order and the launch arguments. '
     'Not called config_file: realsense2_camera declares one of those and '
     'opens it as YAML, and launch configurations reach into included '
     'launch files.'),
    ('grasp_z_offset', '0.010',
     'Added to the detected top of the object to get the grasp height. '
     'Positive: the tool stops above it. Measured grips came in at 14.7 mm '
     'and 25 mm above the detected top, both with the arm stopping short of '
     'its command, so there is real tolerance here -- what the jaws do not '
     'tolerate is being sent below the object. A negative value was what '
     'pressed the tool into the table once the arm started tracking its '
     'trajectory properly.'),
    ('disengage_on_failure', 'true',
     'Take the motors off once a failed cycle has parked the arm somewhere '
     'the collision world says is free. openarm_hardware disables the motors '
     'when its component is deactivated, and these are direct-drive with no '
     'brakes -- so the arm is held up by nothing afterwards. Only ever done '
     'after a refuge is reached, never with the arm stranded over the '
     'table.'),
    ('home_requires_pre_pick', 'true',
     'Whether HOME may be commanded when pre_pick could not be reached on '
     'the way back. HOME is a joint goal to a folded posture, and from a low '
     'extended one the short path in joint space goes through the work '
     'surface -- measured once as a sweep out to x=0.44 and down to z=0.358 '
     'across the object the arm had just failed to pick. On, the retreat '
     'lifts higher, retries pre_pick, then stops and reports.'),
    ('regrasp_attempts', '3',
     'How many times to go back down and try the same grasp when the '
     'detector says the object never moved. The fingers cannot tell a grip '
     'from a fingertip resting on an edge; the object still being where it '
     'was can. Bounded because the same grasp failing the same way three '
     'times will not work on the fourth, and each go costs a descent.'),
    ('fake_hardware', 'false',
     'Whether the arms are mock_components rather than motors. Used for one '
     'thing: deciding whether a saved setting that switches a check off may '
     'be honoured. The rehearsal needs those values and the arms must never '
     'see them, and pick_place_config.json cannot tell which run is reading '
     'it. pick_place_demo.launch.py passes use_fake_hardware straight '
     'through to this.'),
    ('verify_detect_tries', '2',
     'How many times to ask the detector before believing it when it says '
     'there is nothing there. One empty frame is not an answer: at that '
     'moment the arm is hovering directly over the object, which is the one '
     'place guaranteed to hide it from the camera. An empty answer no longer '
     'confirms a grasp either way -- see verify_grasp.'),
    ('contact_torque_margin', '4.0',
     'Stop a move when a joint pulls this many Nm harder than it was pulling '
     'contact_torque_window ago, and back the arm off to where it was '
     'contact_rewind_seconds before. A rolling baseline, and a relative one: '
     'the shoulder carries the whole arm while the wrist carries a gripper, '
     'so one absolute number would either miss a wrist collision or fire '
     'constantly on joint1. Gravity took joint1 from +3.4 to -10.5 Nm across '
     'one measured transit, so the comparison has to be against a moment '
     'ago, not against the start of the move. 0 disables it; fake hardware '
     'reports no efforts, so it never fires there.'),
    ('contact_torque_window', '0.4',
     'How far back the rolling torque baseline looks, seconds.'),
    ('contact_torque_hold', '0.25',
     'How long the excess torque has to last before it counts, seconds. An '
     'acceleration transient passes; something the arm is leaning on does '
     'not.'),
    ('contact_lag_rad', '0.25',
     'How far behind its own trajectory the arm has to fall, rad, before '
     'high torque is read as contact rather than as its own weight. The half '
     'of the test gravity cannot fake. The worst honest tracking error '
     'measured at full speed was 0.126 rad. 0 drops the lag test; the stall '
     'test still applies.'),
    ('contact_rewind_seconds', '2.0',
     'How far back to rewind after a contact stop, seconds.'),
    ('contact_retreat_speed', '0.5',
     'How fast to retrace those seconds on the way back out, rad/s. The one '
     'move in the cycle driven straight at the controller with no collision '
     'check, safe only because every waypoint is a posture the arm measured '
     'itself in moments earlier.'),
    ('preflight_lift', 'true',
     'Include the lift in what the pre-flight proves, so the way out of the '
     'column is checked before the arm goes into it. Without it a descent '
     'can pass and its lift then be refused with the object already gripped '
     'at the bottom of the column -- measured twice, at 2.97 and 3.01 rad '
     'against a 1.5 rad budget. Checked to the pre-grasp height, which is '
     'what lift_column falls back to.'),
    ('descend_recheck', 'true',
     "Re-probe the descent from the arm's measured posture once it has "
     'arrived, rather than trusting where the pre-flight predicted the '
     'approach would end. Measured: the pre-flight passed a column, TRANSIT '
     'landed 32.6 mm off, and the descent then cost 6.42 rad against a 1.5 '
     'rad budget and was refused -- a whole approach flown for a descent '
     'that was never the one checked. One service call per orientation, no '
     'motion.'),
    ('descend_close_gap', 'true',
     'Fly the remainder when the descent stops short of the grasp. The '
     'descent plan ends on the point and the arm does not -- measured 15.5, '
     '29 and 36 mm above it across three runs, plan_error_mm 0.0 every time. '
     'That varies by 20 mm, so grasp_z_offset cannot dial it out: the offset '
     'that grips one run closes on air or presses the table the next. A '
     'continuation of the same descent, straight down the same line. On by '
     'default since run 1789014831, where it stopped short enough to miss: '
     'of four right-arm descents, the two that gripped stopped 3.2 and '
     '4.0 mm from the grasp and the two that missed stopped 18.1 and '
     '18.9 mm out with 15 mm of it height, jaws closing on air above the '
     'object. All tracking error -- at the end of those moves the joints '
     'were still 29 mrad from the last point of their own trajectory, '
     'against 9-11 mrad on the two that worked. The risk it was off for, '
     'pressing into the table, is what the contact guard now watches.'),
    ('descend_gap_max', '0.06',
     'Largest gap, metres, that is treated as tracking error and flown. '
     'Beyond it something else is wrong and the jaws close where they are.'),
    ('grasp_max_depth', '0.0',
     'How far below the detected top of the object the tool may be '
     'commanded, metres. The object rests on the work surface, so its top '
     'face is the only surface reference available without being told where '
     'the table is. This is the floor min_grasp_z is not -- min_grasp_z is '
     'measured from the base, and this table stands at z=0.34. It also '
     'bounds the ladder, whose lower-8mm rungs subtract height on a retry.'),
    ('min_grasp_z', '0.01',
     'Hard floor on grasp height above the base, metres. A sanity clamp on a '
     'bad depth reading -- one came back 0.74 m below the floor -- not a '
     'table floor; grasp_max_depth is that.'),
    ('max_z_clamp', '0.05',
     'How far below min_grasp_z a detection may be and still be clamped, '
     'metres. Further than this and the depth reading is wrong, so x and y '
     'cannot be trusted either and the detection is refused instead.'),
    ('workspace_radius', '1.0',
     'Detections further than this from the base in x-y are refused, metres. '
     'A dead depth stream reports objects metres away; without this the whole '
     'retry ladder is spent collecting IK_FAIL on a point outside the room.'),
    ('max_grasp_z', '0.80',
     'Detections above this are refused, metres -- nothing on the work '
     'surface is that high.'),
    ('check_reach', 'true',
     'Ask /compute_ik whether the grasp, pre-grasp and transit heights are '
     'reachable before moving. An object outside the envelope fails every goal '
     'identically, and no ladder strategy recovers it, so the cycle stops with '
     '"out of reach" instead of six attempts that blame the planner.'),
    ('at_goal_tolerance', '0.02',
     'How close counts as already being at a named posture, radians. A retry '
     're-enters PRE_PICK every attempt; without this the arm re-plans a move '
     'it has already made and visibly shuttles back and forth.'),
    ('use_table_collision', 'false',
     'Add an explicit collision box for the work surface instead of relying on '
     'the octomap alone. Worth enabling with octomap:=static.'),
    ('table_z', '0.0',
     'Height of the work surface top face, world frame. Only used when '
     'use_table_collision is true.'),
    ('velocity_scaling', '0.4',
     'Fraction of joint velocity limits. cuMotion applies min(velocity, '
     'acceleration) as a time dilation of the path it already optimised, so '
     'this changes speed and not the path.'),
    ('acceleration_scaling', '0.4', 'Fraction of joint acceleration limits.'),
    ('detection_reuse_age', '10.0',
     'How old a detection may be and still be picked from without asking the '
     'detector again. choose_arm detects to decide which arm reaches and the '
     'first attempt wanted its own detection a few seconds later -- two '
     'inference waits at the same stationary object, 22.6 s of a measured '
     '76 s cycle. A failed attempt takes nearer a minute, so 10 s reuses the '
     'one and re-detects the other. 0 disables reuse.'),
    ('grasp_finger_min', '0.003',
     'Finger position above which the gripper counts as holding something. '
     'Set to -1.0 to rehearse on fake hardware, where mock_components '
     'reports the finger exactly where it was commanded and the check can '
     'never pass. native/run_pick_place_demo.sh --fake does that for you.'),
    ('object_moved_eps', '0.05',
     'How far the object must have moved for a place to count, metres. Set '
     'to 0.0 on fake hardware, where the object never physically moves so '
     're-detection always finds it back at the pick point.'),
    ('gripper_max_effort', '20.0',
     'max_effort on the GripperCommand goal. Inert on this hardware: the v10 '
     'interface drives the gripper as a position command with a fixed KP and '
     'passes no effort term at all, so nothing reads this and '
     'gripper_torque_cap is what actually bounds the grip. Left at the '
     "controller's own stall-detection value. It was 2.0 here against 20.0 in "
     'the declaration, which the launch silently won -- a contradiction worth '
     'not leaving in place even where it changes nothing.'),
    ('gripper_torque_cap', '2.5',
     'Grip limit as torque at the gripper motor, in Nm -- the same units as '
     'the exoskeleton bridge. 2.5 Nm is about 59.5 N at the finger over the '
     '42.0 mm/rad transmission. The panel sets this at runtime and the close '
     're-reads it, so it applies to the next grasp without a restart.'),
    ('refresh_octomap_at_home', 'true',
     'Refresh the octomap at HOME, and nowhere else. HOME is out of the '
     'camera view; anywhere the arm can be seen from, it gets captured as an '
     'obstacle sitting exactly where it is about to plan from.'),
    ('home_pose_tolerance', '0.05',
     'How close the measured joints must be to HOME, radians, before the map '
     'may be captured.'),
    ('home_settle_tolerance', '0.20',
     'How far a *stopped* joint may stand off HOME and still count as '
     'arrived, radians. For the joint that cannot reach its commanded value '
     'at all -- see home_joint_positions. Without it the cycle failed with '
     '"could not reach home" while the arm was sitting at home.'),
    ('joint_still_speed', '0.05',
     'Speed below which a joint counts as stopped, rad/s. Separates "as '
     'close as this hardware gets" from "still on its way".'),
    ('joint_limit_margin', '0.02',
     'Keep commanded postures this far off the position limits, radians. A '
     'goal on a limit cannot be held, and gives cuRobo no room on that '
     'joint.'),
    ('approach_frame', 'tool',
     'How the move above the object is aimed. "tool" solves the full '
     'hand_tcp pose here and sends the chosen solution as a joint goal. '
     '"wrist" solves for joint7\'s centre with joint7 left out and tilts it '
     'afterwards -- measured within a few thousandths of a radian of "tool" '
     'once that tilt is accounted for, 0.172 against 0.175 at the pre-grasp, '
     'so it buys no headroom and costs a separately pinned grasp yaw. '
     '"planner" sends a pose goal and lets cuMotion pick the configuration, '
     'which is what this did before -- and what left joint3 and joint5 '
     'sitting on their stops, where no straight line can continue. What does '
     'the work in either of the first two is *choosing* the solution instead '
     'of taking the first one offered.'),
    ('grasp_jaw_flip', 'true',
     'Also try the grasp with the jaws turned 180 degrees. A parallel gripper '
     'closes on the same two faces either way round, so it is the same grasp '
     '-- but a very different posture: 0.000 rad of joint headroom one way, '
     '0.172 the other.'),
    ('grasp_yaw_free', 'false',
     'Leave the grasp yaw to the solver instead of pinning it to the '
     "object's axis. Keep it off for anything that is not round: the jaws "
     "close along the tool frame's y, which is link6's y-axis, and joint7 "
     'turns about that very axis so it cannot move it -- an unconstrained '
     'solve therefore picks the closing angle arbitrarily. Measured: a '
     'straight-line descent onto a point 6.3 mm from target, then the '
     'gripper shutting to 0 mm at 1.14 Nm against a 2.5 Nm cap.'),
    ('approach_tilt_stage', 'false',
     'Send the joint7 tilt as its own move once the arm is over the object. '
     'Off by default: the tool hangs 180.1 mm off that joint, so tilting last '
     'swings it through a 116 mm arc immediately before the descent, and '
     'costs a second plan. The wrist partition is in the solve -- which still '
     'leaves joint7 out -- and does not have to be copied by the execution.'),
    ('grasp_tilt_max', '0.35',
     'How far off vertical the gripper may come down, radians. A strictly '
     'top-down grasp is six constraints on seven joints at a fixed point, and '
     'near the edge of the envelope there is often no solution clear of the '
     'joint stops at all -- measured, all 25 of them against a limit. 0.35 is '
     '20 degrees, which on most objects grips just as well. Vertical is tried '
     'first and tilts in increasing order, so nothing tilts that need not; 0 '
     'restores the strict behaviour.'),
    ('grasp_tilt_steps', '2',
     'Tilt magnitudes to try between 0 and grasp_tilt_max.'),
    ('grasp_tilt_azimuths', '4',
     'Directions to try each tilt in: 4 is away, left, toward, right.'),
    ('use_grasp_model', 'true',
     "Take the grasp from GraspNet when it has one, instead of synthesising "
     'a top-down pose from the detector box. It answers what the synthesised '
     'pose only guesses at -- where on the object, at what height, in which '
     'direction, and how far the jaws must open -- all read off the point '
     'cloud. Whether the arm can get there stays the pre-flight\'s job. '
     'Falls back to the synthesised grasp whenever nothing usable is '
     'proposed, which is common: GraspNet was trained for a 100 mm gripper '
     'and this one spans 44 mm. Needs grasp_model:=true to have anything to '
     'listen to.'),
    ('grasp_model_max_age', '30.0',
     'How old a candidate list may be, seconds, before it is treated as '
     'being about some earlier look.'),
    ('grasp_model_max_offset', '0.08',
     'How far a candidate list\'s object may be from the one being picked, '
     'metres, before it is treated as being about something else.'),
    ('request_grasps', 'true',
     'Ask the grasp server for candidates on reaching the staging pose. Only '
     'a topic publish -- nothing acts on the answer yet -- so it is free to '
     'leave on, and it is how the candidates get looked at on real data '
     'before anything is wired to them.'),
    ('grasp_yaw_options', '3',
     'Alternative grasp yaws to try when nothing that respects the '
     "detector's axis estimate flies, spread over 180 degrees (half a turn "
     'is the jaw flip, already covered). Tried last: a tilt still grips the '
     'correct faces, a yaw change is a different grasp, and on a screwdriver '
     'the measured axis is right. Measured: a roll of tape the pre-flight '
     'refused outright flew its whole column at 90 degrees and at no other '
     'yaw. 0 treats the axis as fixed.'),
    ('arm_choice_cheap', 'true',
     'Choose the arm on KDL alone, taking "cannot tell" as "use the half of '
     'the frame the object is in". The solvers behind KDL exist to '
     'second-guess its "no" and are where the time goes -- 2.9 s per '
     'orientation per height per arm, measured, whether they find anything '
     'or not, which was 52 s before the robot moved. The pre-flight settles '
     'whether the pick is possible for whichever arm is chosen.'),
    ('reach_orientations', '10',
     'Orientations the reach probe may try per point. It sends a planning '
     'request for each, so the full tilt set would make an unreachable object '
     'cost a minute of probing; the pre-flight explores the rest.'),
    ('posture_margin', '0.10',
     'Joint-limit headroom, radians, an approach posture should have. A '
     'threshold rather than a score: past it, more headroom buys nothing and '
     'the ranking prefers the posture nearest the staging pose instead -- one '
     'with 0.493 rad of headroom that the arm had to reconfigure across the '
     'workspace to reach left TRANSIT sitting for 20 seconds with the tool '
     'not moving. If nothing clears it the roomiest is used anyway; the '
     'pre-flight is the real gate.'),
    ('posture_seeds', '48',
     'Random restarts per posture solve. About a second, once per pick '
     'attempt.'),
    ('single_descent', 'true',
     'One descent instead of two. The cycle used to stop at the pre-grasp, '
     'open the gripper there and descend again -- two lines to solve, two '
     'settles, two offset corrections and a visible pause in mid-air, for a '
     'stop nothing needed. With this the gripper opens above the object and a '
     'single straight line goes all the way to the grasp.'),
    ('check_planner_ready', 'true',
     "Before the cycle, ask the planner to plan a goal to the arm's own "
     'current posture. Plan-only, so nothing moves, and it cannot fail for '
     'reasons about the target -- which is the difference between "the '
     'planner is not planning" and a report blaming the recorded pre-pick '
     'pose.'),
    ('linear_transit', 'true',
     'Fly the long move above the object as a straight line as well -- the '
     'same motion as dragging the end-effector arrow in RViz. Pre-flighted '
     'like the rest: the line is planned from the staging posture and the '
     'descent legs from its end, so it is only used when the whole column '
     'still flies from where the line leaves the arm. Everything below the '
     'transit was already a Cartesian line; this leg was the last joint-space '
     'move between the staging pose and the grasp.'),
    ('preflight_descent', 'true',
     'Prove the descent before the arm leaves its rest pose. Candidate '
     'postures come from the kinematic chain and /compute_cartesian_path '
     'takes an explicit start state, so "does a straight line down solve from '
     'the posture I intend to be in" is answerable without moving. Without '
     'it, the cycle found out at the pre-grasp -- gripper already open, 18% '
     'of the line solvable -- and went back to pre-pick and home to start '
     'again.'),
    ('preflight_candidates', '6',
     'Postures the pre-flight may probe. Each costs up to four '
     '/compute_cartesian_path calls and no motion, which trades a few seconds '
     'before moving against minutes of failed attempts after.'),
    ('retry_after_preflight', 'false',
     'Run the retry ladder even after a pre-flight has passed. Off by '
     'default: the next rung finds the same geometry, so the only visible '
     'effect is the arm travelling back to pre-pick and home for another '
     'identical attempt. Stochastic planner failures are resent in place by '
     'plan_attempts either way.'),
    ('pose_tolerance', '0.005',
     'How close the *tool* has to end up, metres. A move MoveIt calls SUCCESS '
     'has satisfied the joint controller tolerance, which is a different '
     'thing: the tool was landing 13.7 mm low at z=0.405 and 33.5 mm low at '
     'z=0.555, in the direction gravity pulls.'),
    ('pose_abort_limit', '0.15',
     'How far out the tool may settle before a leg counts as failed, metres. '
     'Not an accuracy standard -- legs here routinely settle 25 to 52 mm out '
     'and still pick the object up, and the one successful grasp landed '
     '32.2 mm from its commanded point. This is the distance past which the '
     'tool is somewhere else entirely: a left-arm descent reported '
     'fraction=1.0 and settled 287 mm from the target, and was recorded as '
     'ok. 0 disables the check.'),
    ('pose_settle_time', '4.0',
     'Seconds to let the tool creep onto its target before judging it. The '
     "servo's integral term is slow -- held at one target the error went 19.3 "
     'to 10.7 mm over about eleven seconds -- so waiting is what closes it. '
     'Commanding the same target again does not move the arm at all.'),
    ('pose_offset_correction', 'false',
     'After settling, aim past the target once by the error still remaining. '
     'Off. It was built for an error that was repeatable and almost purely '
     'vertical (dz -13.8, -14.0, -14.0, -14.0 mm across four passes) and the '
     'error is no longer that: measured since, it improved two transits and '
     'made one transit and three descents worse. On the approach it also does '
     'harm beyond its own leg -- aiming past the transit point commanded a '
     'position 13.7 mm higher and 17.6 mm out in y, so the descent started '
     'from somewhere the pre-flight had not checked and had to travel '
     'sideways as well as down, ending 52.7 mm out. A few millimetres above '
     'the object is harmless; the descent is a fresh line to the grasp.'),
    ('descend_linear_only', 'true',
     'Refuse the descent and the retreat when no straight line can be had, '
     'rather than substituting free-space hops. The hops are not a milder '
     'version of the same motion: a descent whose line solved 37.5% -- '
     'identically checked and unchecked, so the arm runs out of reach along it '
     '-- became three hops that swung the tool 3.7 cm sideways and aborted '
     'with CONTROL_FAILED against the table.'),
    ('motion_sample_period', '0.1',
     'How often to sample the arm during a move, seconds, for the motion log. '
     'Endpoints alone cannot tell a straight descent from one that swings '
     'sideways on the way. 0 keeps only the endpoints.'),
    ('motion_log_heartbeat', '1.0',
     'Seconds between heartbeat records while a cycle runs. Without them a '
     'stall is a silent gap in the file and there is no telling a wedged '
     'planner from an arm crawling somewhere. 0 disables them.'),
    ('cartesian_partial_min', '0.5',
     'Smallest share of a straight line worth flying. Above this it is '
     'executed and the remainder requested as another line, so the whole move '
     'stays straight. Refusing partials was wrong: a leg whose line solved '
     '95.65% was thrown away for curved hops that aborted against the table.'),
    ('cartesian_segments', '4',
     'How many straight segments one leg may take.'),
    ('cartesian_min_gain', '0.002',
     'Least distance a segment must gain, metres, before another is tried. A '
     'line that stalls in the same place is stalling against something, and '
     'pushing further walks the gripper into it.'),
    ('motion_sample_limit', '40',
     'Most samples kept per motion; beyond this the middle is thinned and the '
     'ends preserved.'),
    ('motion_log', 'motion_log.jsonl',
     'Append-only record of every commanded motion, one JSON object per line: '
     'what was asked for, which mechanism ran, what came back, and the joint '
     'positions, velocities and motor efforts before and after. Relative paths '
     'are under the workspace. Empty disables it.'),
    ('auto_start', 'false',
     'Start a cycle 3 s after launch instead of waiting for '
     '/pick_place/start.'),
]


def generate_launch_description():
    prompt = LaunchConfiguration('prompt')
    model_id = LaunchConfiguration('model_id')

    declarations = [
        DeclareLaunchArgument(
            'prompt', default_value='detect screwdriver',
            description='PaliGemma detection prompt for the object to pick.'),
        DeclareLaunchArgument(
            'model_id', default_value='google/paligemma-3b-pt-224',
            description='HuggingFace model id for the detector.'),
        DeclareLaunchArgument(
            'inference_mode', default_value='on_demand',
            description='on_demand runs the detector only after a prompt '
                        'arrives, which is what a pick needs. continuous is '
                        'the old behaviour: a full generate every 0.4 s for '
                        'ever, holding 6.4 GB and the GPU with it.'),
        DeclareLaunchArgument(
            'idle_unload_after', default_value='20.0',
            description='Seconds idle before the weights move off the GPU. '
                        'Coming back costs about a second. 0 keeps them '
                        'resident.'),
    ]
    declarations += [
        DeclareLaunchArgument(name, default_value=default, description=description)
        for name, default, description in ORCHESTRATOR_ARGS
    ]

    # PYTHONUNBUFFERED because launch pipes stdout, which makes Python
    # block-buffer it: without this the detector's "model loaded" and the
    # orchestrator's state lines only reach the console and launch.log in 8 KB
    # chunks, so a run that fails early looks like it printed nothing at all.
    unbuffered = {'PYTHONUNBUFFERED': '1'}

    detector = ExecuteProcess(
        cmd=[
            os.path.join(WS, 'VLM', 'run_vlm_detector.sh'),
            '--ros-args',
            '-p', ['prompt:=', prompt],
            '-p', ['model_id:=', model_id],
            '-p', ['inference_mode:=', LaunchConfiguration('inference_mode')],
            '-p', ['idle_unload_after:=',
                   LaunchConfiguration('idle_unload_after')],
        ],
        name='vlm_detector',
        output='screen',
        additional_env=unbuffered,
    )

    orchestrator_cmd = [
        'python3', os.path.join(WS, 'pick_place_orchestrator.py'),
        '--ros-args',
        '-p', ['prompt:=', prompt],
    ]
    for name, _default, _description in ORCHESTRATOR_ARGS:
        orchestrator_cmd += ['-p', [f'{name}:=', LaunchConfiguration(name)]]

    orchestrator = ExecuteProcess(
        cmd=orchestrator_cmd,
        name='pick_place_orchestrator',
        output='screen',
        additional_env=unbuffered,
    )

    return LaunchDescription(declarations + [detector, orchestrator])
