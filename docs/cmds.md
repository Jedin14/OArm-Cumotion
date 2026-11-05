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
#   click the picture to pick a point; Move, Detect box, Pick all, Recapture, Return, STOP (Esc)
#   Bypass (on by default): Detect box picks the box at once when it can be picked -- no Move to press
#   Pick all: every box in view, one after another (detect, pick, drop, next) -- ONE try per box: a box
#     it cannot pick, or that the camera says is still in its place, is skipped (flagged when the cup touched
#     it; Forget skipped brings them back; one DROPPED off its place is not); the base is never asked to
#     move; STOP ends it. To clear all it can: when no box is left but some were skipped and others
#     taken meanwhile, it goes round again (up to 3 rounds; boxes out of reach for good are skipped at once,
#     no reach check), until a round takes none. Once the arm (and the box on the cup) is out of the
#     camera picture on its way to the drop, it checks the pick and finds + plans the next box meanwhile,
#     so the arm goes straight on when it is back
#   Pick tab: navigation / pre-pick, front / top down, teach, ArUco ext, tool lengths
#   Advanced tab: Vacuum On / Off by hand (any time; Off drops a box on the cup), box picker settings, typed target
#   Calibrate tab: live markers with ids, Cal arms, Overlay, Anchor + Refit, camera Calibrate + Apply
#   System tab: the ROS topics (it serves the same arm_state / arm_action as run_pick_stack.sh)
#   options: port:=8090 host:=127.0.0.1 model:=false approach:=top order:=v2 score:=0.4
#            walls_edges:=depth|picture walls_ahead:=0.11  (the green walls where the camera cannot see:
#            from where the DEPTH ends, 11 cm in front of the nearest thing seen; picture / 0.05 = as before
#            2026-11-05: more reach at the sides, but objects in the depth-less bands are invisible)
#            drop_velocity:=0.7 (pre_pick -> drop with the box on, 0..1; was 0.4) correct_speed:=0.04
#            (the correction moves, m/s; was 0.02) drop_hold:=1.0 (s at the drop, vacuum off)
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
