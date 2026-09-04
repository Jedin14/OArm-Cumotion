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
    ('home_joint_positions', '[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]',
     'HOME, joint1..joint7 in radians: where the arm rests, observes from, '
     'and returns to, and the only pose the octomap is captured from. Seven '
     'zeros is the "home" group state in the SRDF, which folds the arm out '
     'of the camera view. Jog somewhere else and call '
     '/pick_place/capture_ready for its numbers.'),
    ('arm_selection', 'by_side',
     '"by_side" chooses the arm before moving: the half of the camera frame '
     'the object is in is preferred, and whichever arm can actually reach it '
     'gets the job. "fixed" always uses `arm`, for single-arm work.'),
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
    ('transit_height', '0.20',
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
    ('grasp_z_offset', '-0.005',
     'Added to the detected surface height to get the grasp height.'),
    ('min_grasp_z', '0.01', 'Hard floor on grasp height, metres.'),
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
    ('velocity_scaling', '0.3',
     'Fraction of joint velocity limits. cuMotion applies min(velocity, '
     'acceleration) as a time dilation of the path it already optimised, so '
     'this changes speed and not the path.'),
    ('acceleration_scaling', '0.3', 'Fraction of joint acceleration limits.'),
    ('gripper_max_effort', '2.0',
     'Grip force limit in newtons -- finger_joint1 is prismatic, so its effort '
     'is a force. NOTE: the v10 hardware interface currently discards it, '
     'which is why gripper_torque_cap exists; see the README.'),
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
