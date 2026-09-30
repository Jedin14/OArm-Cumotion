"""Everything, in one command: robot, planners, detector, orchestrator, panel.

    native/run_pick_place_demo.sh

That wrapper is the intended entry point -- it checks the things that fail
confusingly rather than clearly (no DISPLAY, no /workspaces symlink, CAN down)
before any of this starts. To run the launch file directly:

    ros2 launch pick_place_demo.launch.py

What it starts, and why in this order:

    launch_everything.launch.py   MoveIt, RViz, controllers, cuMotion, the
                                  camera, the octomap gater. tool_frame is
                                  forced to match `arm`, because cuMotion takes
                                  Cartesian goals for exactly one link and a
                                  mismatch makes every pose goal fail.
    pick_place.launch.py          the PaliGemma detector and the orchestrator.
    pick_place_ui.py              the panel you type into.

The detector loads several GB of PaliGemma weights, which takes tens of seconds;
it starts alongside the robot rather than after it so that load overlaps with
bringup. Until it prints "model loaded" a pick will sit in LOCATE. This is the
reason the two launch files are normally kept separate -- you do not want that
load on every robot bringup -- so use pick_place.launch.py on its own when the
robot is already up.

Before the first run, record the two poses the cycle needs:

    python3 record_states.py
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression

WS = os.path.dirname(os.path.realpath(__file__))

# Passed straight through to pick_place.launch.py, which declares the rest.
FORWARDED = ['arm', 'arm_selection', 'prompt', 'states_file', 'place_mode',
             'pick_place_config',
             'approach_height',
             'velocity_scaling', 'acceleration_scaling', 'gripper_max_effort',
             'detection_reuse_age', 'grasp_finger_min', 'object_moved_eps',
             'grasp_hold_torque_min', 'grasp_width_drop',
             'place_prompt', 'place_clearance', 'place_max_drop',
             'place_speed',
             'place_after_pick', 'boot_walk', 'boot_pose', 'boot_speed',
             'locate_from_staging', 'boot_other_arm', 'grasp_table_clearance',
             'boot_via_home', 'recover_after_contact', 'carry_guard',
             'use_home',
             'grasp_z_offset', 'grasp_max_depth', 'descend_close_gap',
             'descend_gap_max', 'home_requires_pre_pick',
             'disengage_on_failure', 'disengage_on_contact',
             'refresh_octomap_at_home', 'map_before_pick',
             'planner_probe_timeout',
             'use_table_collision', 'table_z']


def generate_launch_description():
    arm = LaunchConfiguration('arm')

    declarations = [
        DeclareLaunchArgument(
            'arm', default_value='right',
            description='Which arm picks. Also decides cuMotion\'s tool_frame.'),
        DeclareLaunchArgument(
            'arm_selection', default_value='by_side',
            description='"by_side" takes the half of the camera frame the '
                        'object is in -- fast, and right nearly every time. '
                        '"by_reach" asks the solvers which arm can reach '
                        'first, which is slow. "fixed" always uses arm.'),
        DeclareLaunchArgument(
            'prompt', default_value='detect screwdriver',
            description='Initial detector prompt. The panel overrides it.'),
        DeclareLaunchArgument(
            'states_file', default_value='auto',
            description='Poses recorded by record_states.py. "auto" resolves '
                        'to pick_place_states_<arm>.yaml.'),
        DeclareLaunchArgument(
            'place_mode', default_value='detected',
            description='Where the object is released: state, home, position '
                        'or detected.'),
        DeclareLaunchArgument(
            'place_prompt', default_value='piece of paper',
            description='place_mode "detected": what the drop surface is '
                        'called. The object is put down on top of whatever '
                        'the detector finds for this.'),
        DeclareLaunchArgument(
            'place_max_drop', default_value='0.010',
            description='The most the object may be above the sheet when '
                        'the jaws open, metres. The place refuses rather '
                        'than dropping from higher.'),
        DeclareLaunchArgument(
            'place_clearance', default_value='0.005',
            description='The gap left under the object when it is let go, '
                        'metres. How far it hangs below the tool comes '
                        'from what the pick measured.'),
        DeclareLaunchArgument(
            'place_speed', default_value='0.2',
            description='Speed cap for the descent onto the drop surface '
                        'and the retreat off it.'),
        DeclareLaunchArgument(
            'use_home', default_value='false',
            description='Whether HOME may be commanded at all. Off: the '
                        'staging pose is the rest pose, the observation '
                        'pose and where a cycle returns to, and nothing '
                        'drives to HOME.'),
        DeclareLaunchArgument(
            'carry_guard', default_value='true',
            description='Carrying something back to the staging pose, keep '
                        'the tool out of the other arm\'s half. The plan is '
                        'measured before it is flown, because the planner '
                        'takes no path constraints and the other arm is '
                        'parked over there with nothing in the map to say '
                        'so.'),
        DeclareLaunchArgument(
            'place_after_pick', default_value='false',
            description='Whether one press of Pick also places. Off: Pick '
                        'ends holding at the staging pose and Place puts it '
                        'down.'),
        DeclareLaunchArgument(
            'boot_walk', default_value='true',
            description='At bringup, walk both arms to boot_pose and wait '
                        'there.'),
        DeclareLaunchArgument(
            'boot_pose', default_value='pre_pick',
            description='Where they wait: "pre_pick", the staging pose, or '
                        '"home". The boot walk maps the work area from '
                        'HOME on its way either way.'),
        DeclareLaunchArgument(
            'use_rviz', default_value='true',
            description='Start RViz. false on a headless or VNC session: '
                        'without hardware GL it hangs at start-up with its '
                        'window never mapped, and nothing else needs it.'),
        DeclareLaunchArgument(
            'boot_via_home', default_value='false',
            description='Whether the boot walk folds down to HOME to map '
                        'before going to where the arms wait. Off: '
                        'straight there, and the map is taken from there.'),
        DeclareLaunchArgument(
            'recover_after_contact', default_value='true',
            description='After a torque trip: motors off, let it settle, '
                        'motors on, glide to the staging pose.'),
        DeclareLaunchArgument(
            'boot_other_arm', default_value='pre_pick',
            description='Where the arm that is not picking waits: "home", '
                        '"pre_pick" or "leave". HOME keeps it out of the '
                        'picking arm\'s way.'),
        DeclareLaunchArgument(
            'grasp_table_clearance', default_value='0.005',
            description='How close to the measured surface the jaws may '
                        'close. The floor that stops a deeper grasp '
                        'pressing into the table.'),
        DeclareLaunchArgument(
            'locate_from_staging', default_value='true',
            description='Whether a cycle starting at the staging pose looks '
                        'from there rather than going to HOME first.'),
        DeclareLaunchArgument(
            'boot_speed', default_value='0.15',
            description='Speed cap for the boot walk, as a share of the '
                        'joint limits.'),
        DeclareLaunchArgument(
            'approach_height', default_value='0.05',
            description='Pre-grasp height above the object, metres.'),
        DeclareLaunchArgument(
            'velocity_scaling', default_value='0.4',
            description='Fraction of joint velocity limits. Time-scales the '
                        'planned path, so it costs tracking margin not accuracy.'),
        DeclareLaunchArgument(
            'acceleration_scaling', default_value='0.4',
            description='Fraction of joint acceleration limits.'),
        DeclareLaunchArgument(
            'gripper_max_effort', default_value='20.0',
            description='max_effort on the GripperCommand goal. Inert on this '
                        'hardware -- gripper_torque_cap is what bounds the '
                        'grip. This layer had 2.0 against 20.0 in '
                        'pick_place.launch.py, and this layer is the one the '
                        'demo script uses.'),
        DeclareLaunchArgument(
            'refresh_octomap_at_home', default_value='true',
            description='Capture the octomap only at HOME, never at READY -- '
                        'the arm is in the camera frame at READY.'),
        DeclareLaunchArgument(
            'map_before_pick', default_value='true',
            description='Clear the octomap and retake it at the start of '
                        'every pick, so the plan is made against the table '
                        'as it is now rather than as the boot walk left it.'),
        DeclareLaunchArgument(
            'planner_probe_timeout', default_value='10.0',
            description='Seconds the pre-flight planner probe waits for '
                        'move_group before reporting it wedged.'),
        DeclareLaunchArgument(
            'use_table_collision', default_value='false',
            description='Explicit collision box for the work surface.'),
        DeclareLaunchArgument(
            'table_z', default_value='0.0',
            description='Work surface height, world frame.'),
        DeclareLaunchArgument(
            'collision_activation_distance', default_value='0.03',
            description='Clearance cuMotion keeps from every obstacle, metres. '
                        'cuRobo default is 0.01.'),
        DeclareLaunchArgument(
            'octomap', default_value='static',
            description='live or static. static plus the gater is the tested path.'),
        DeclareLaunchArgument(
            'use_fake_hardware', default_value='false',
            description='Rehearse without the arms. Grasp verification cannot '
                        'pass on fake hardware -- see the README.'),
        DeclareLaunchArgument(
            'right_can_interface', default_value='can0',
            description='CAN interface for the right arm.'),
        DeclareLaunchArgument(
            'left_can_interface', default_value='can1',
            description='CAN interface for the left arm.'),
        DeclareLaunchArgument(
            'ui', default_value='false',
            description='Start the old Tk panel as well. The web UI '
                        'supersedes it; this is here for a machine with no '
                        'browser to hand.'),
        DeclareLaunchArgument(
            'grasp_model', default_value='false',
            description='Start the GraspNet server, which publishes ranked '
                        '6-DoF grasps on /grasp/candidates. Off by default: '
                        'it is third-party, noncommercial-licensed, and has '
                        'to be fetched with grasp/fetch_graspnet.sh first. '
                        'Publishing is harmless -- nothing acts on it unless '
                        'use_grasp_model is also on.'),
        DeclareLaunchArgument(
            'web', default_value='true',
            description='Start the web UI. Reachable from any machine on the '
                        'network, which is the point -- the workspace lives '
                        'on the robot and is driven from elsewhere.'),
        DeclareLaunchArgument(
            'web_port', default_value='8088',
            description='Port the web UI listens on.'),
        DeclareLaunchArgument(
            'pick_place_config', default_value='pick_place_config.json',
            description='Where the editable sequence and the UI-settable '
                        'parameters are saved. Read at bringup, so a run '
                        'starts where the last one left off.'),
        DeclareLaunchArgument(
            'detection_reuse_age', default_value='10.0',
            description='How old a detection may be and still be picked from '
                        'without asking the detector again. 0 disables '
                        'reuse.'),
        DeclareLaunchArgument(
            'grasp_finger_min', default_value='0.003',
            description='Finger position above which the gripper counts as '
                        'holding something. -1.0 to rehearse on fake '
                        'hardware.'),
        DeclareLaunchArgument(
            'grasp_hold_torque_min', default_value='1.0',
            description='Torque at the gripper motor, Nm, below which the '
                        'jaws count as slack rather than gripping. 0 turns '
                        'the torque half of the grasp check off.'),
        DeclareLaunchArgument(
            'grasp_width_drop', default_value='0.002',
            description='How much narrower than the close the jaws may sit '
                        'and still count as holding the same thing, metres. '
                        'Catches an object that slipped out afterwards.'),
        DeclareLaunchArgument(
            'object_moved_eps', default_value='0.05',
            description='How far the object must have moved for a place to '
                        'count, metres. 0.0 to rehearse on fake hardware.'),
        DeclareLaunchArgument(
            'grasp_z_offset', default_value='-0.030',
            description='Added to the *top* of the object to get the grasp '
                        'height. Negative reaches down the side of it; '
                        'grasp_table_clearance is the floor, so it cannot '
                        'reach through the surface.'),
        DeclareLaunchArgument(
            'grasp_max_depth', default_value='0.0',
            description='How far below the detected top of the object the '
                        'tool may be commanded, metres. The floor '
                        'min_grasp_z is not.'),
        DeclareLaunchArgument(
            'descend_close_gap', default_value='true',
            description='Fly the remainder when the descent stops short of '
                        'the grasp -- straight down the same line, as a '
                        'continuation of the one descent. On since run '
                        '1789014831, where the two descents that missed '
                        'stopped 18 mm out with 15 mm of it height and the '
                        'jaws closed on air; the two that gripped stopped '
                        'within 4 mm.'),
        DeclareLaunchArgument(
            'disengage_on_failure', default_value='false',
            description='Take the motors off after a failed cycle has parked '
                        'the arm somewhere checked. Off: an ordinary failure '
                        'leaves a healthy arm, and releasing it costs a '
                        'restart of the stack. See disengage_on_contact and '
                        'the Stop button.'),
        DeclareLaunchArgument(
            'disengage_on_contact', default_value='true',
            description='Take the motors off after the contact guard fires '
                        '-- the arm has run into something and is leaning '
                        'on it.'),
        DeclareLaunchArgument(
            'home_requires_pre_pick', default_value='true',
            description='Refuse HOME when pre_pick could not be reached on '
                        'the way back, rather than sweeping the arm across '
                        'the work surface to get there.'),
        DeclareLaunchArgument(
            'descend_gap_max', default_value='0.06',
            description='Largest gap, metres, treated as tracking error and '
                        'flown rather than reported.'),
    ]

    # One link, one arm: openarm_<arm>_hand_tcp. Deriving it here rather than
    # exposing it means the two cannot disagree.
    tool_frame = PythonExpression(["'openarm_' + '", arm, "' + '_hand_tcp'"])

    robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(WS, 'launch_everything.launch.py')),
        launch_arguments={
            'tool_frame': tool_frame,
            'octomap': LaunchConfiguration('octomap'),
            'collision_activation_distance':
                LaunchConfiguration('collision_activation_distance'),
            'use_fake_hardware': LaunchConfiguration('use_fake_hardware'),
            'use_rviz': LaunchConfiguration('use_rviz'),
            'right_can_interface': LaunchConfiguration('right_can_interface'),
            'left_can_interface': LaunchConfiguration('left_can_interface'),
        }.items(),
    )

    pick_place = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(WS, 'pick_place.launch.py')),
        launch_arguments=dict(
            {name: LaunchConfiguration(name) for name in FORWARDED},
            # Under a different name at each end, so it cannot be forwarded
            # by the list above: the robot layer calls it use_fake_hardware
            # and the orchestrator calls it fake_hardware. It has to get
            # there, though -- it is what stops a rehearsal's saved
            # grasp_finger_min=-1.0 being honoured on the arms.
            fake_hardware=LaunchConfiguration('use_fake_hardware'),
        ).items(),
    )

    # Its own process and its own environment, like the detector: the torch
    # the CUDA extensions were compiled against is not cuRobo's.
    grasp = ExecuteProcess(
        cmd=[os.path.join(WS, 'grasp', 'run_grasp_server.sh')],
        name='grasp_server',
        output='screen',
        additional_env={'PYTHONUNBUFFERED': '1'},
        condition=IfCondition(LaunchConfiguration('grasp_model')),
    )

    web = ExecuteProcess(
        cmd=['python3', os.path.join(WS, 'pick_place_web.py'),
             '--port', LaunchConfiguration('web_port')],
        name='pick_place_web',
        output='screen',
        additional_env={
            'PICK_PLACE_CONFIG': LaunchConfiguration('pick_place_config'),
            'PYTHONUNBUFFERED': '1',
        },
        condition=IfCondition(LaunchConfiguration('web')),
    )

    ui = ExecuteProcess(
        cmd=['python3', os.path.join(WS, 'pick_place_ui.py')],
        name='pick_place_ui',
        output='screen',
        additional_env={
            'PICK_PLACE_STATES_FILE': LaunchConfiguration('states_file'),
            'PYTHONUNBUFFERED': '1',
        },
        condition=IfCondition(LaunchConfiguration('ui')),
    )

    return LaunchDescription(
        declarations + [robot, pick_place, grasp, web, ui])
