# Everyday commands (on beast)

native/run_launch_everything.sh   # terminal 1: robot, MoveIt, cuMotion, camera; arms boot to navigation

native/run_auto_box.sh            # terminal 2: AUTO BOX
#   PICK MODE (p)    both arms to pre_pick, the box model loads
#   DETECT BOX (space)  the box model finds the target box: bounding box, x y z, PICKABLE or not
#   PICK (m)         the arm goes to the box (same motion as click_to_move), then pre_pick -> drop -> pre_pick
#   NAVIGATION (n), TOP DOWN (d), RETURN (b); tool row: L/R TOOL on/off, [ ] and , . for length

# AUTOMATIC CALIBRATION (web GUI, Calibrate tab; sheets at http://beast:8080/sheets/calib_box_sheets.pdf
#   and /sheets/charuco_board.pdf, or: python3 -m ctm.calib_box):
#   1 Depth check: hold the A4 board in view -> Depth: capture; move it; 3-5 times -> Depth: fit
#   2 Cal tip: the 170x100x25 marker box on the cup (cup on the cross of face B), arm at pre-pick -> Cal tip (box)
#     ~15 min unattended; reports the tip error on postures it did not fit (mm); the old model is kept as arm_model.before_boxcal_*.yaml
native/run_web_gui.sh             # or: EVERYTHING in one browser page -> http://beast:8080 (from the laptop)
#   click the picture to pick a point; Move, Detect box, Auto pick, Recapture, Return, STOP (Esc)
#   Pick tab: navigation / pre-pick, front / top down, teach, ArUco ext, tool lengths
#   Calibrate tab: live markers with ids, Cal arms, Overlay, Anchor + Refit, camera Calibrate + Apply
#   System tab: the ROS topics (it serves the same arm_state / arm_action as run_pick_stack.sh)
#   options: port:=8090 host:=127.0.0.1 model:=false approach:=top
#   native/run_web_gui.sh --click     # click-only, like click_to_move: no box model, no Detect box / Auto pick
#   (run only one of web GUI / pick stack / the OpenCV windows at a time: they all drive the arms)

native/run_pick_stack.sh          # or: the ROS 2 module (box model + arms), window only if there is a display
#   options: gui:=auto|on|off  approach:=auto|top|front  model:=true|false
#   everything is under /beast_arm:
#   ros2 topic pub --once /beast_arm/arm_action std_msgs/msg/String "{data: ready_to_pick}"   (or "ready_to_pick top vacuum", "stop"; vacuum = left arm only, finger = right arm only)
#   ros2 topic echo /beast_arm/arm_action/status   # detecting, target_found, picking, checking, pick_complete / pick_unverified: / no_target: / pick_failed: / stopped
#   ros2 topic echo /beast_arm/pick_result --full-length   # the camera's pick check: picked true/false/null, outcome (success, missed, moved, ...)
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
#   (after openarm.yml changes -- the tool / held-box spheres -- restart run_launch_everything.sh)

native/run_click_to_move.sh       # or: click a point, MOVE (TEACH (t): nudge a touch onto the spot, SAVE)

native/run_calibrate.sh           # CALIBRATION only (the folded pieces on both grippers):
#   live camera view, every ArUco marker outlined with its id
#   CAL ARMS (f)     arm model, front + top-down: marker 11 on the wall 50 cm in front;
#                    confirm the overlay; ~35-40 min; each arm back to pre_pick after its turn
#   CALIBRATE (c)    camera pose from the arms (after moving the camera); APPLY writes it
#   OVERLAY (o)      robot model over the camera image (a check)
#   click + MOVE     test a point: touches it (tool row = tool length) and goes back to pre_pick
#   (the pieces are in the planners here only; the other tools take them out)

python3 -m ctm.board --sheet      # regenerate gripper_boards.pdf (print at 100 %, fold on the dashed lines)
python3 -m ctm.board --live       # quick camera view of the markers
