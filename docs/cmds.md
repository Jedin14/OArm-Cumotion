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
#   Tests (domain 77): scratchpad trim_test.py (4), selffilter_test.py (5), filter_test.py (8), lidar_test3.py (13).

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
