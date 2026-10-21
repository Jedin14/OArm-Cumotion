# Everyday commands (on beast)

native/run_launch_everything.sh   # terminal 1: robot, MoveIt, cuMotion, camera; arms boot to navigation

native/run_auto_box.sh            # terminal 2: AUTO BOX
#   PICK MODE (p)    both arms to pre_pick, the box model loads
#   DETECT BOX (space)  the box model finds the target box: bounding box, x y z, PICKABLE or not
#   PICK (m)         the arm goes to the box (same motion as click_to_move), then pre_pick -> drop -> pre_pick
#   NAVIGATION (n), TOP DOWN (d), RETURN (b); tool row: L/R TOOL on/off, [ ] and , . for length

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
#   ros2 topic pub --once /arm_action std_msgs/msg/String "{data: ready_to_pick}"   (or "ready_to_pick top")
#   ros2 topic echo /arm_action/status   # detecting, target_found, picking, pick_complete / no_target: / pick_failed:
#   ros2 topic pub --once /arm_state/cmd std_msgs/msg/String "{data: pre_pick}"     (navigation, return, status)
#   ros2 topic echo /arm_state           # navigation, pre_pick, moving_to_*, detecting, picking, arm_out, ...
#   ros2 service call /arm_state/get std_srvs/srv/Trigger
#   ros2 topic echo /pick_target         # pick_interfaces/PickTarget: position, approach top/front, pixel box
#   (rebuild the messages after a change: native/build_ws.sh --packages-select pick_interfaces)

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
