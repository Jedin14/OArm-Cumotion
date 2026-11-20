# Everyday commands (on beast)

native/run_launch_everything.sh   # terminal 1: robot, MoveIt, cuMotion, camera; arms boot to navigation
#   the camera's accelerometer (the box picker's tilt from gravity) is on by itself once its motion
#   sensor is readable -- the RealSense udev rules, once, on beast (then unplug and replug the camera):
#     sudo wget -O /etc/udev/rules.d/99-realsense-libusb.rules https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules
#     sudo udevadm control --reload-rules && sudo udevadm trigger
#     ls -l /dev/iio:device*      # readable by you now (before: crw------- root only)
#   accel:=true / accel:=false force it; the GUI's Box picker card shows TILT ... ACCEL or TF
#   the VACUUM RELAY starts here too (app/vacuum_bridge.py: /beast_arm/vacuum -> the ESP32 by its USB id,
#   /dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_34:CD:B0:16:1D:30-if00, ttyACM0 now):
#   vacuum:=false leaves it out, vacuum_port:=... another port; its lines in this terminal: [vacuum_bridge-N]
#   the 4D LIDAR (Unitree L2, 2026-11-11) starts here too: YOUR driver (4D_lidar's l2_driver, with
#   4D_lidar/install/.../l2.yaml; only the driver, not the stabilizer) and robot/lidar_bridge.py. The L2
#   is in serial mode on /dev/serial/by-id/usb-1a86_USB_Single_Serial_593A033327-if00. The driver spins
#   it up (~25 s from standby to the first cloud) and back to standby at Ctrl-C. Don't run
#   4D_lidar/scripts/l2.sh at the same time (one driver per L2; terminal 1 says so and leaves it out).
#   Mount: 111.3 mm above the centre of the arms' bases, 2.8 mm in front: TF openarm_body_link0 ->
#   l2_lidar FITTED TO THE CAMERA on 2026-11-12 (its cloud laid onto the D455's, the window cleaned;
#   agreement 30% -> 54%): at (-0.0518, 0.0452, 0.8408) -- 6.7 cm behind, 3.9 cm left of the tape
#   measure (0.0154, 0.0062, 0.8457) -- blue (the dome) forward and 21 deg below level, red up and
#   forward leaning left, green to the right: lidar_x:=-0.0518 lidar_y:=0.0452 lidar_z:=0.8408
#   lidar_pitch:=110.9 lidar_yaw:=-1.3 lidar_spin:=161.3 (spin 180 = red straight up-forward). The software takes that mount as set: it cannot
#   aim the L2 -- it must physically sit so. (lidar_tilt:=imu
#   takes the tilt from its accelerometer instead; on 2026-11-12 that read 1.9 g at rest: not trusted.)
#   ONLY TRUE POINTS are shown and used (RViz "LiDAR L2", "Merged camera + LiDAR", the octomap): none
#   while the LiDAR fails a check (then the camera alone); none while the L2 reports its WINDOW DIRTY
#   (chip LIDAR ✕ · dirty window; /lidar/status dirty_pct, limit 15 (wiped it sits at 8-11), held up to 17 so it does not flicker -- 2026-11-12 it was 18%: a shell of
#   false points 0.3-0.9 m in front of it; clean the dome); never a point the camera sees through.
#   The driver's raw cloud stays on /l2/cloud.
#   The CAMERA CHECKS THE MOUNT: the LiDAR's points in the camera's picture must sit at the camera's
#   depth (30%, 4 cm). Upright or tilted/turned otherwise, they don't: the chip says LIDAR ✕ · mount?
#   and the walls stay. lidar_spin:= (deg, default 160) turns it about its own Z
#   (blue in RViz, the dome axis) after the tilt: which side faces where (turned round: +180). Live:
#   ros2 param set /lidar_bridge mount_spin_deg 161.3 / mount_rpy_deg "[0.0, 110.9, -1.3]" (also mount_xyz). RViz: "LiDAR L2" (/lidar/cloud) and "Merged camera + LiDAR" (/merged_cloud,
#   coloured); its work-area cloud goes into the octomap with the camera's (sensors_3d.yaml, l2_lidar).
#   While it works (LIDAR ✓ · walls off) the green walls are off; it stops, is missing or fails the
#   camera's check: the walls are back at the next capture (--lidar-walls always: walls whatever).
#   Args: lidar:=serial (default) | udp | off; lidar_port:= ; lidar_x:= lidar_y:= lidar_z:= (m);
#   lidar_roll:= lidar_pitch:= lidar_yaw:= lidar_spin:= (deg). Its state: ros2 topic echo --once /lidar/status
#   RViz looks in the frame 'level' (gravity straight down): the robot's own frame leans ~5 deg against
#   the room (camera accelerometer 5.3, ceiling in the L2's cloud 4.6). Args level_pitch:=5.3 level_roll:=-0.2
#   (deg; 0 0 = as before). Planning stays in 'world'. No picture in RViz: Fixed Frame back to world.
#   SINCE 2026-11-13 (robot/lidar_bridge.py; all three on by default, each a parameter):
#   - self_filter: the robot is never an object. Points on the arms / hands / body (the URDF's
#     collision shapes at the live TF, body_pad 8 cm for the LiDAR, body_pad_camera 3 cm for the
#     camera) and anything on the way from the LiDAR to them are out of /lidar/cloud,
#     /lidar/cloud_work and /merged_cloud. /lidar/status: self_filter ("on"), robot_points.
#     (MoveIt's own self-filter in sensors_3d.yaml still guards the octomap, 12 cm.)
#   - camera_first: where the camera has a depth it shows that surface (mm-exact) and the
#     LiDAR's points there are dropped; the LiDAR stays where the camera cannot see.
#   - steady_filter: a LiDAR point stays only if its range repeats along its view line over
#     the last 2 s (steady_min 4 points, at most half as many beyond). /lidar/status: unsteady_pct.
#   Measured 2026-11-13 with the box 0.9 m away (dirty 17%): 69% of the L2's points in the
#   camera's view were false; the steady filter keeps 93% of the true and 25% of the false.
#   The L2 near a big box is a +-3 cm instrument at best: the camera does the picking.
#   - range_trim: the L2's ranges are a few cm off, the same at 0.6 and 1.0 m but changing with
#     its window (+1.2 cm at 10% dirty, -2.7 cm at 17%). Where the camera sees the same surface it
#     is right: once a second the median difference is taken off ALL the LiDAR's ranges (settles in
#     ~6 s, at most 5 cm). /lidar/status: range_trim_cm (applied), range_left_cm (what is left).
#   - density_filter: no specks -- a LiDAR point goes on only with >= 8 others within 3 cm over the
#     last 2 s (replay 2026-11-16: isolated work-space voxels 134 -> 52). The gater also removes
#     small islands of camera depth (cv2.filterSpeckles, speckle_px 200, speckle_diff_mm 25).
#   Tests (domain 77): scratchpad speck_test.py (2), trim_test.py (4), selffilter_test.py (5), filter_test.py (8), lidar_test3.py (13).

# SENSOR ALIGN (2026-11-16): move the camera's and the LiDAR's world onto the robot, live, by hand
#   (terminal 1 running; a window on beast's desktop; nothing moves the arm):
#     source native/setup.bash && python3 robot/sensor_align.py
#   The yellow ball in RViz ("Sensor align tip", /sensor_align/markers) is where the robot thinks the
#   working tip is. Hold your palm on the real tip and move the clouds until the palm sits on the ball.
#   Move: Both (camera + LiDAR together) / Camera / LiDAR; buttons move the cloud along X/Y/Z (0.5-20 mm)
#   and turn it about the ball (0.05-2 deg). Fit camera to the arm: its depth onto the arm's model, no
#   motion (a shift only from one posture; it says which way it could not see). Fit LiDAR to the
#   camera: the L2's mount onto what the camera sees (it says how firm). Readouts: camera vs arm (%
#   within 15 mm), LiDAR vs camera. Undo / Revert / Rebuild octomap / Save (camera -> v10.urdf.xacro +
#   cam_org.txt + rebuild; LiDAR -> the lidar_* defaults here). Unsaved: gone at the next terminal 1.
#   After moving the camera, run the Auto calibrate again (the arm model saw the old camera).
#   The L2's HOUSING box (MoveIt obstacle l2_lidar_housing) stays where the L2 physically is
#   (lidar_bridge housing_xyz / housing_rpy_deg / housing_spin_deg, default the 11-13 fit), not at the
#   aligned mount: on 11-16 a by-hand mount 6.4 cm left put the box into openarm_left_link0 (painted red).
#   Test (domain 77, synthetic room): scratchpad align/run_test.sh.

# AUTOMATIC CALIBRATION (web GUI, Calibrate tab; sheets at http://beast:8080/sheets/calib_box_sheets.pdf
#   and /sheets/charuco_board.pdf, or: cd app && python3 -m ctm.calib_box):
#   1 Depth check: hold the A4 board in view -> Depth: capture; move it; 3-5 times -> Depth: fit
#   2 Cal tip: the 170x100x25 marker box on the cup (cup on the cross of face B), arm at pre-pick -> Cal tip (box)
#     ~15 min unattended; reports the tip error on postures it did not fit (mm); the old model is kept in
#     old_version/calibration_backups/arm_model.before_boxcal_*.yaml
#   (results: calibration/ -- arm_model.yaml, armcal_anchors.yaml, touch_corrections.yaml, payload_left.yaml,
#    camera_calibration.yaml, depth_correction.yaml; raw recordings: calibration/armcal_samples/)
# the arm poses (robot/pick_place_states_<arm>.yaml), recorded by hand with the arm put there:
#   cd robot && python3 record_states.py --arm left pre_pick_top_state   # the pose after a TOP pick:
#   near pre_pick, the tool pointing straight down (its tool position is where top picks carry the box)
#   RViz's Goal State list (pre_pick, pre_pick_top, drop, navigation, ...) is a copy in
#   src/openarm_ros2/openarm_bimanual_moveit_config/config/openarm_bimanual.srdf: after re-recording,
#   copy the joints there, then native/build_ws.sh --packages-select openarm_bimanual_moveit_config
#   and restart run_launch_everything.sh
native/run_web_gui.sh             # or: EVERYTHING in one browser page -> http://beast:8080 (from the laptop)
#   click the picture to pick a point; Move, Detect box, Recapture, Return, STOP (Esc); Pick all has its own tab
#   Base correction (Pick and Pick all tabs): for a box it cannot pick, the SHORTEST wheel moves into the zone
#     where that pick works (from the logs' reach checks) -- TOP 38-47 cm ahead, 2 cm right..6 cm left
#     (top_near:= top_far:= top_right:= top_left:=), FRONT 44-50 cm, 6..20 cm left; in the zone and still
#     out of reach: to the zone's middle --
#     only the moves needed ("move left 12 cm"); the base is never driven from here
#   Pick all tab, "No base movement" (on by default): a box out of reach is skipped. Off: Pick all stops
#     at it with an alert (the moves, live as the base moves; also <ns>/base_correction, JSON), and picks
#     it by itself once the box is in place and the base has stopped; Try now / Skip box in the alert
#   Bypass (on by default): Detect box picks the box at once when it can be picked -- no Move to press
#   Pick all: every box in view, one after another (detect, pick, drop, next) -- ONE try per box: a box
#     it cannot pick, or that the camera says is still in its place, is skipped (flagged when the cup touched
#     it; Forget skipped brings them back; one DROPPED off its place is not); the base is never asked to
#     move; STOP ends it. To clear all it can: when no box is left but some were skipped and others
#     taken meanwhile, it goes round again (up to 3 rounds; boxes out of reach for good are skipped at once,
#     no reach check), until a round takes none. Once the arm (and the box on the cup) is out of the
#     camera picture on its way to the drop, it checks the pick and finds + plans the next box meanwhile,
#     so the arm goes straight on when it is back
#     Each box's time and the total: the "Pick all" card under the buttons, the log, and
#     native/logs/pick_all/pick_all_<date_time>.json + .csv; when it ends, a banner + chime + the tab's title
#   Pick tab: navigation / pre-pick, front / top down, teach, ArUco ext, tool lengths
#   Advanced tab: Vacuum On / Off by hand (any time; Off drops a box on the cup), box picker settings, typed target
#   Calibrate tab: live markers with ids, Cal arms, Overlay, Anchor + Refit, camera Calibrate + Apply
#   System tab: the ROS topics (it serves the same arm_state / arm_action as run_pick_stack.sh)
#   options: port:=8090 host:=127.0.0.1 model:=false approach:=top order:=v2 score:=0.4
#            walls_edges:=depth|picture walls_ahead:=0.11  (the green walls where the camera cannot see:
#            from where the DEPTH ends, 11 cm in front of the nearest thing seen; picture / 0.05 = as before
#            2026-11-05: more reach at the sides, but objects in the depth-less bands are invisible)
#            held_velocity:=0.4 (the way back out beside the boxes with the box on, 0..1)
#            top_x_offset:=-0.015 (TOP DOWN: goes down 1.5 cm nearer the robot in x, straight ahead; 0 = as measured)
#            drop_velocity:=0.7 (pre_pick -> drop with the box on, 0..1; was 0.4) correct_speed:=0.02
#            (the correction moves, m/s; 0.04 was less accurate) drop_hold:=1.0 (s at the drop, vacuum off)
#            top_faces:=false (the box picker's face rule, box_picker/src/pick_rules.py: a box on a box
#            by its FRONT, its top only if no front is found -> "TOP ONLY" in the log; true = the larger of
#            front and top anywhere, as before 2026-11-05; also on the Advanced tab, "Face of a box on a box")
#   native/run_web_gui.sh --click     # click-only, like click_to_move: no box model, no Detect box / Pick all / Bypass
#   (run only one of web GUI / pick stack / the OpenCV windows at a time: they all drive the arms)

native/run_pick_stack.sh          # or: the ROS 2 module (box model + arms), window only if there is a display
#   options: gui:=auto|on|off  approach:=auto|top|front  model:=true|false  order:=v1|v2  score:=0.4  walls_edges:=depth|picture  walls_ahead:=0.11
#   everything is under /beast_arm:
#   ros2 topic pub --once /beast_arm/arm_action std_msgs/msg/String "{data: ready_to_pick}"   (or "ready_to_pick top vacuum", "stop"; vacuum = left arm only, finger = right arm only)
#   ros2 topic echo /beast_arm/arm_action/status   # detecting, target_found, picking, checking, pick_complete / pick_unverified: / no_target: / pick_failed: / stopped
#   ros2 topic echo /beast_arm/pick_result --full-length   # the camera's pick check: picked true/false/null, outcome (success, missed, moved, ...)
#   ros2 topic echo /beast_arm/base_adjust          # pick_interfaces/BaseAdjust (out of reach, or 2 failed picks on one box): direction left/right/none + distance_m, sideways only
#   ros2 topic echo /beast_arm/arm_action/phase    # each step: "left approach", "left contact", "left back_out", "left to_drop", ...
#   ros2 topic pub --once /beast_arm/arm_state/cmd std_msgs/msg/String "{data: pre_pick}"     (navigation, return, status)
#   ros2 topic echo /beast_arm/arm_state           # navigation, pre_pick, moving_to_*, detecting, picking, arm_out, ...
#   ros2 service call /beast_arm/arm_state/get std_srvs/srv/Trigger
#   ros2 topic pub --once /beast_arm/held_object/size pick_interfaces/msg/ObjectSize "{height_mm: 170, width_mm: 100, depth_mm: 25}"
#   ros2 topic echo /beast_arm/held_object/state   # none / "left 170x100x25 mm" (attached at contact, released at the drop)
#   ros2 topic pub --once /beast_arm/held_object/release std_msgs/msg/Empty
#   ros2 topic pub --once /beast_arm/stop std_msgs/msg/Empty
#   ros2 topic echo /beast_arm/status --full-length   # everything as one JSON object
#   ros2 topic echo /beast_arm/pick_target         # pick_interfaces/PickTarget: position, approach top/front, tool + arm, face_mm, box_mm (the known size matched to the face)
#   (rebuild the messages after a change: native/build_ws.sh --packages-select pick_interfaces)
#   (after robot/openarm.yml changes -- the tool / held-box spheres -- restart run_launch_everything.sh)

native/run_vacuum_bridge.sh       # the VACUUM RELAY on its own -- only after run_launch_everything.sh vacuum:=false
#   (terminal 1 runs it otherwise): /beast_arm/vacuum (std_msgs/Bool) -> the ESP32, '1' on / '0' off over USB serial
#   port:=/dev/serial/by-id/usb-...  (default: our ESP32 by id; auto: the only one there is -- with several it waits)
#   baud:=115200  resend:=1.0 (the state written again every 1 s: an ESP32 that restarts is switched back)
#   ros2 topic echo /beast_arm/vacuum --qos-durability transient_local    # what it follows
#   (restarting it while a box hangs on the cup can drop it: the ESP32 restarts when the port opens)

native/run_click_to_move.sh       # or: click a point, MOVE (TEACH (t): nudge a touch onto the spot, SAVE)

native/run_calibrate.sh           # CALIBRATION only (the folded pieces on both grippers):
#   live camera view, every ArUco marker outlined with its id
#   CAL ARMS (f)     arm model, front + top-down: marker 11 on the wall 50 cm in front;
#                    confirm the overlay; ~35-40 min; each arm back to pre_pick after its turn
#   CALIBRATE (c)    camera pose from the arms (after moving the camera); APPLY writes it
#   OVERLAY (o)      robot model over the camera image (a check)
#   click + MOVE     test a point: touches it (tool row = tool length) and goes back to pre_pick
#   (the pieces are in the planners here only; the other tools take them out)

(cd app && python3 -m ctm.board --sheet)   # regenerate docs/sheets/gripper_boards.pdf (print at 100 %, fold on the dashed lines)
(cd app && python3 -m ctm.board --live)    # quick camera view of the markers

## STOP WHEN CTRL-C HANGS (2026-11-17)
If terminal 1 keeps printing "user interrupted with ctrl-c (SIGINT) again, ignoring...",
run this from another terminal on beast (asks every stack node to stop, then kills
what is left after 10 s, including the hung `ros2 launch`):

    bash /workspaces/isaac_ros-dev/robot/stop_everything.sh

(It stops only the stack's own programs, this user's, on ROS domain 0 -- not an editor,
`less` or `tail` on one of their files, nor a test on another domain.)

## LIDAR: CAMERA FIRST, HOLES AND 12 CM BEHIND (2026-11-17)
lidar_bridge drops the L2's points wherever the camera speaks for that place: in front
of a camera surface, up to camera_first_behind (0.12 m) behind it, in camera holes up to
camera_first_fill_px (25 px) from a depth, and up to 25 px outside the picture. Reason:
the L2 sat 4-8 cm behind the camera's box faces and its scatter in the camera's edge
holes made voxels 3-4 cm in front of a box, blocking every way in. In a hole and outside
the picture (a depth the camera did not measure there) only the L2 points AT the borrowed
surface go: at most camera_first_fill_front (0.05 m) in front of it -- anything further in
front is something only the L2 sees (thin, dark, shiny, too near) and stays (review fix,
2026-11-17 evening). As before (live, no restart): ros2 param set /lidar_bridge
camera_first_behind 0.04 ; ros2 param set /lidar_bridge camera_first_fill_px 0
The LiDAR mount itself is ~4 cm off the camera (median), range trim stuck at +5 cm:
re-fit it with robot/sensor_align.py (Fit LiDAR to the camera, several arm postures).
#   2026-11-17: Fit LiDAR to the camera reads /lidar/cloud_fit (the bridge's cloud NOT filtered by
#   the camera -- /lidar/cloud lacks exactly the points on the camera's surfaces); restart terminal 1
#   once so the bridge publishes it (an older bridge: the fit falls back to /lidar/cloud, biased).

## PICK ALL: LOOK ONCE, THEN THE ORDER (2026-11-18)
The picker's first look gives the whole grid order (box_model/scene "plan", world coords).
After each pick, with the arm out of the picture, ONE depth frame is compared with the last
look (inside the stack, outside the picked box): unchanged -> the next box of the order is
reach-checked and picked at once (no DETECT BOX); a change elsewhere (>15 mm over a 600 px
patch or 2 % of the stack) -> a full look again. A box the cup weighed is "picked", one it
weighed empty is not -- no camera pick check (it waited ~30 s, "arm still in view", per box).
Off switches (web_gui args): --pick-all-plan off   --pick-all-verify camera
Tune: --plan-change-mm 15 --plan-change-px 600. --approach-settle now 1.5 s (was 2.5).

## MOTOR DROP-OUT, PICK ALL IN VIEW, LIDAR KEEP-OUT (2026-11-18)
- A motor that stops driving (status 1 -> 0, or 9 / 13): the driver holds it at its measured
  position, re-enables it every 0.2 s (max 25), then eases it back to the command at 0.25 rad/s
  (recovery.hpp; motor_gains_*.yaml auto_reenable / reenable_rate). Real faults (8, 10-12, 14)
  are not retried. The app STOPs everything at once (--stop-on-motor-drop on); PICK ALL ends
  saying which motor. Cause seen 11-18 10:48: left joint2, no fault code, during USB "EMI?"
  drop-outs -- a brownout/noise restart. Check its power and CAN cable.
- PICK ALL never asks the picker's camera check (it waited ~30 s with the hand in view): the
  cup's weight, else one depth frame of the box's spot. Full looks only with the arm back at
  pre_pick; a check spoiled by the arm is redone at once when it is back.
- l2_lidar_keepout: the LiDAR housing box grown 12 cm up, 8 cm back, 5 cm right, 1 cm left /
  front (lidar_bridge keepout_pad; all zero: none). >= 16 mm clear of the arm bases / body.
  REPLACED 2026-11-19 by the square box below (keepout_pad is only the fallback now).

## L2 LIDAR MODEL AND KEEP-OUT BOX (2026-11-19)
- RViz draws the real L2 on the robot: link l2_lidar_body in v10.urdf.xacro (fixed on
  openarm_body_link0 at housing_xyz, dome tilted forward 113.4 deg), mesh
  src/openarm_description/meshes/sensors/unitree_l2.stl, made from Unitree's L2 user manual
  drawing (75 x 75 x 65 mm). Drawn only: no collision in the URDF (cuMotion uses
  robot/openarm.urdf). Moved L2? Change the joint origin there, then
      native/build_ws.sh --packages-select openarm_description      (and restart terminal 1)
- l2_lidar_keepout is now a see-through ORANGE box square to the robot (MoveIt + cuMotion both
  plan round it): x -55..100, y -55..65, z 815..1000 mm in openarm_body_link0 -- 6 cm in front
  of the L2, 8 cm above it, 1-2 cm behind / to the sides; 20 mm above both arm bases, 42 mm
  above the body. Of 453 poses the arms really went to, it blocks only the 11-18 10:45 approach
  that passed 24 mm from the L2 (next nearest: 6.9 mm clear).
  While it holds the L2, the green l2_lidar_housing box is taken out of the scene (it only hid
  the model); it comes back if the keep-out is turned off.
      ros2 param set /lidar_bridge keepout_box "[-0.055, 0.12, -0.055, 0.065, 0.815, 1.0]"  # resize
      ros2 param set /lidar_bridge keepout_box "[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"   # off (-> keepout_pad)
      ros2 param set /lidar_bridge keepout_rgba "[1.0, 0.55, 0.0, 0.3]"           # colour / see-through
  (applied within 3 s). RViz: MotionPlanning -> Scene Geometry -> Show Scene Geometry on.

## IMU AND BOX SIZE TOPICS, PICK ALL CAMERA CHECK (2026-11-18)
- /camera/camera/imu        sensor_msgs/Imu: D455 gyro + accelerometer (unite_imu_method 2); the
                            separate /camera/camera/accel/sample and /gyro/sample stay.
                            launch_everything gyro:=false turns the gyro (and /imu) off.
- /l2/imu                   sensor_msgs/Imu from l2_driver (frame l2_imu); its accelerometer reads
                            ~1.9 g at rest (the L2's own scale) -- not corrected here.
- /beast_arm/box_model/box_size   pick_interfaces/ObjectSize (latched): the target box as measured,
                            mm, no margin (depth 0 = not seen). Every detection.
- /beast_arm/box_model/boxes      std_msgs/String JSON (latched): the target and every box in the
                            order -- xyz (world), approach, cell, face_mm, box_mm {w, h, d}.
  ros2 topic echo /beast_arm/box_model/box_size
- PICK ALL pick check: the cup's weight AND the camera (the box's spot in one depth frame, arm out
  of the picture) after every pick; the weight wins, a disagreement is said; weighed empty but the
  spot emptied = "dropped". --pick-all-verify camera: the picker's own check as before.

## OBSTACLE BOX UNDER THE ARMS (2026-11-19)
- robot/height_box.py (terminal 1 starts it; height_box:=off to leave it out): planning-scene box
  `height_box`, 20 (front-back) x 60 (left-right) x 28 (deep) cm, centred under the arms' centre,
  its top 152 - (x + 10) cm below it, x from /c60000004/vertical_distance; 100 cm while nothing
  has come in for 3 s.
  2026-11-19 later: it goes UP ROUND THE STAND (sent in pieces round the stand's URDF shape, layer
  by layer); only the camera / LiDAR / arm bases hold it down, 1.5 cm under the lowest over it
  (the camera: top <= 0.549 m, x <= ~123.5 cm). Arm links inside it: status arm_in_box (and the
  slider window says ARM IN THE BOX) -- the planners will not start from there.
      ros2 topic echo /height_box/status --qos-durability transient_local
- Test (domain 77): scratchpad test_height_box.py (7 checks, no TF) + test_height_box2.py (6: round
  the stand, held by the camera, arm in the box).

## GUARD ROUND THE CAMERA (2026-11-19)
- robot/camera_guard.py (terminal 1; camera_guard:=off to leave it out): planning-scene object
  `camera_guard`, see-through orange plates round the D455: 20 mm in front (7 mm off the face),
  15 mm at the ends, over the front lip, under both ends. Not one box: the camera sits half in the
  URDF body column, and a box would put the robot in collision with itself. It follows the camera TF,
  and a plate within 3 mm of the body or the arm bases is left out (and logged).
- Before it, cuMotion did not know the camera at all (robot/openarm.urdf has none), and MoveIt
  knew it with no margin. 11 recorded left-arm moves passed 8-15 mm from it (forearm past its left
  end). The guard blocks 12 of 455 recorded poses, all within ~27 mm of the camera.
      ros2 topic echo /camera_guard/status --qos-durability transient_local
      ros2 param set /camera_guard rgba "[1.0, 0.55, 0.0, 0.3]"
- Test (domain 77): scratchpad test_camera_guard.py (5 checks); geometry: cam_guard_eval.py.

## STAND-IN /c60000004/vertical_distance PUBLISHER (2026-11-19)
- For trying the height box without the device (not while the device publishes: two sources):
      ros2 launch robot/vertical_distance.launch.py                     # SLIDER window: x 0..152 cm
                       (0 = sensor right above the box; shows the gap and what height_box says;
                       Fixed / Sweep / Off; gui:=off for no window; slider_min_cm / slider_max_cm)
      ros2 launch robot/vertical_distance.launch.py value_cm:=30        # gap 152-(30+10) = 112 cm
      ros2 launch robot/vertical_distance.launch.py mode:=sweep min_cm:=20 max_cm:=60 period_s:=20
      ros2 launch robot/vertical_distance.launch.py type:=string        # "x cm" text instead of the
                       default sensor_msgs/Range (range = x/100 m, frame c60000004_range; x outside
                       min_range_m..max_range_m 0..1.52 -> +-inf = no reading)
      python3 robot/vertical_distance_pub.py --ros-args -p value_cm:=30 # the same without launch
  While running:
      ros2 param set /vertical_distance_pub value_cm 50
      ros2 param set /vertical_distance_pub mode sweep        # fixed / sweep / off (off: height_box -> 100 cm)

## NO DROP POSE WHILE THE BOX BELOW IS HIGH (2026-11-19)
- While x (/c60000004/vertical_distance, sensor_msgs/Range) reads 87-152 cm, a picked box is let go
  at PRE_PICK: the arm goes straight back to pre_pick, the vacuum goes off there (--drop-hold s),
  the box is taken out of the planners -- never the drop pose (it would collide). Outside that range,
  or no reading for 3 s: the drop pose as before. Said in the log: "x .. cm - the drop pose would
  collide; letting go at pre_pick instead".
      --no-drop-x 87 152      (web GUI / pick stack / click_to_move; "--no-drop-x 0 0": off)
      --vertical-topic /c60000004/vertical_distance
- Test: scratchpad test_no_drop.py (12 checks: 86.9/152.1 drop, 87/100/152 pre_pick, stale, none,
  off, top pick).

## PRESS STOPPED SHORT / DROP POSE BLOCKED (2026-11-19 17:36 fix)
- The first try at 17:36 never reached the box: each press stopped "at the torque limit" (1.0 Nm, the
  vacuum tool's soft limit) 26, 23 and 6 mm into a 31 mm press -- in free air. The shoulder's stick-slip
  swings 0.6-1.2 Nm in 0.5 s, and the soft limit was armed over the whole press (EARLY_CONTACT_M 3 cm;
  the press starts ~3 cm out). 17 of the last 50 recorded presses stopped short like that (11-18, 11-19
  10:13, 10:23, 17:20, 17:36); the next try then "worked" unchanged, by luck.
  Now (app/ctm/robot.py guarded_line): the tool's 1.0 Nm limit at / past the measured surface as before,
  before it only while the tip is not keeping up (< half the commanded speed over 0.35 s: blocked or
  squashing); the arm's 2.0 Nm limit everywhere; contact (stuck + torque rise) armed from 1.5 cm before
  the surface (was 3 cm). Replayed on the 50 recorded presses: the 17 false stops go, the 7 real ones
  at / past the surface unchanged. A box nearer than measured still stops at 1.0 Nm (the tip blocks).
- The drop pose is checked before going (MoveIt's collision check, held box included): in collision
  -> the box is let go at pre_pick, said why. 17:36 / 17:38: at x 84.5 (under --no-drop-x 87) the hand
  was IN the height box; cuMotion tried 3 x 11 s + OMPL (~58 s) and you switched the vacuum off by hand.
- (Replaced 11-20: "weighed nothing -> no drop trip" -- the weighing misses boxes on the cup.)
- Tests: app/tests/ (see below).

## IS THE BOX ON THE CUP? THE CAMERA DECIDES, NOT THE WEIGHING (2026-11-20)
- 11-19 17:36 the cup weighed "nothing" (-36, +4, +5 g) with the box on it: a FRONT-picked box still
  rests on the one below while it is weighed (3 cm out), so the shoulder torques barely see it.
- Now: before the arm moves, one depth frame of the box's spot (the face round the pick point); once
  the arm is back at pre_pick (out of the picture -- checked: the cup and a hanging box are off the
  left edge), the spot again:
    still there (60 % unchanged)  -> nothing taken: no drop trip, the box released from the planners
                                     (unless the cup weighed a box: then the drop, to be safe)
    gone (60 % farther/no reading)-> on the cup: to the drop, even if the weighing said nothing
    cannot tell (hidden, no frames)-> the drop, as always
  The weighing alone never skips the drop any more. --spot-check off: as before (~0.3 s per pick).
- Never left hanging / waiting:
    the drop pose in collision, or x 87-152 cm   -> let go at pre_pick (11-19)
    the drop leg does not plan                   -> one try only (was 3 x 11 s + OMPL), back at
                                                    pre_pick and LET GO there (it used to keep the
                                                    box on the cup)
    no way home                                  -> 2 more tries by itself (via the home posture,
                                                    then with a held box let through the map)
                                                    before "press RETURN"
- The 4D LiDAR is not used for this: it can see the cup at pre_pick, but lidar_bridge drops every point
  within 8 cm of the arm, it is +-3 cm, and it scatters false points near surfaces -- a 54 mm box on
  the cup is not reliably told from air. The sure way: a vacuum pressure sensor on the ESP32.
- Tests (kept in the repo now; the scratchpad is wiped on reboot):
      source native/setup.bash
      PYTHONNOUSERSITE=1 python3 app/tests/test_drop_decisions.py    # 26: drop / let go / spot / retries
      PYTHONNOUSERSITE=1 python3 app/tests/test_press_cap.py          # the guarded press, stick-slip
      python3 app/tests/replay_presses.py                             # the 50 recorded presses

## VACUUM ON THE TOP ROW: SQUARE TO THE FACE OR A BASE MOVE (2026-11-20)
- Why the cup missed the centre on the top row: at shoulder height (box z 0.71, shoulders 0.73) and
  49 cm ahead, the left arm only reached the face with the tool 22-36 deg off square ("tilted 20
  turn -15/-30"). The square ways in there sit at the very edge of reach (no room for the camera's
  last correction) and were dropped. A cup that comes in that oblique touches rim-first, slides off
  the centre as it squashes, and does not hold: the top row 2 of 11 picked today (it weighed 75 g and
  67 g on the cup, then the box stayed in its spot); the one that came in square (12:55) worked.
  Lower rows: square ways in (median 5-7 deg). The face centre itself is not off (the outline is
  6-9 mm big on every row alike).
- Now: --vacuum-face-max 20 (deg; FRONT, tool on): no way in more than 20 deg off square to the face
  (top-down picks keep --vacuum-lean). A top-row box with none -> the base correction, which for a
  box at --high-row-z 0.65 or higher aims at --front-zone-high 0.55 0.60 0.05 0.30 (box 55-60 cm
  ahead, 5-30 cm left) instead of --front-zone 0.44 0.50 0.06 0.20.
- Reach maps (the real reach check on the URDF, deg off the face with room; app/tests/reach_map.py Z):
      z 0.44 (bottom): square 40-50 cm ahead, oblique or none at 55-60
      z 0.58 (middle): square 40-60 cm ahead
      z 0.71 (top):    square only 55-60 cm ahead (45-50: 20-23 deg)
  So one stand-off rarely suits the bottom and the top rows: the base moves back for the top row.
- Test: PYTHONNOUSERSITE=1 python3 app/tests/test_vacuum_face.py   (10, incl. the real reach check)
  To go back: --vacuum-face-max 90 (and --front-zone-high = --front-zone).
