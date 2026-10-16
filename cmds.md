# Everyday commands (on beast)

native/run_launch_everything.sh   # terminal 1: robot, MoveIt, cuMotion, camera; arms boot to navigation

native/run_auto_box.sh            # terminal 2: AUTO BOX
#   PICK MODE (p)    both arms to pre_pick, the box model loads
#   DETECT BOX (space)  the box model finds the target box: bounding box, x y z, PICKABLE or not
#   PICK (m)         the arm goes to the box (same motion as click_to_move), then pre_pick -> drop -> pre_pick
#   NAVIGATION (n), TOP DOWN (d), RETURN (b); tool row: L/R TOOL on/off, [ ] and , . for length

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
