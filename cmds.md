# Everyday commands (on beast)

native/run_launch_everything.sh   # terminal 1: robot, MoveIt, cuMotion, camera; arms boot to navigation

native/run_auto_box.sh            # terminal 2: AUTO BOX
#   PICK MODE (p)    both arms to pre_pick, the box model loads
#   DETECT BOX (space)  the box model finds the target box: bounding box, x y z, PICKABLE or not
#   PICK (m)         the arm goes to the box (same motion as click_to_move), then pre_pick -> drop -> pre_pick
#   NAVIGATION (n), TOP DOWN (d), RETURN (b); tool row: L/R TOOL on/off, [ ] and , . for length

native/run_click_to_move.sh       # or: click a point, MOVE; and the calibrations:
#   CALIBRATE (c)    camera pose from the arms (after moving the camera)
#   CAL ARMS (f)     arm model: 2 calibration pieces per gripper (gripper_boards.pdf), marker ID 11
#                    on a wall 50 cm in front (docs/gripper_markers.pdf); confirm the overlay; ~20 min
#   TEACH (t)        nudge the tip onto the spot after a touch; taught on top of the arm model

python3 -m ctm.board --sheet      # regenerate gripper_boards.pdf (print at 100 %)
