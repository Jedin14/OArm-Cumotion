VLM/run_in_vlm_env.sh vlm_detect.py "screwdriver" --once --source realsense
native/run_click_to_move.sh    # click a point + MOVE: fingertips touch it. CALIBRATE (automatic) once after moving the camera (after native/run_launch_everything.sh)
#   FLAGS / f : calibrate the gripper flags (print gripper_markers.pdf at 100%, IDs 10/11 left, 20/21 right).
#               Clear ~50 cm in front of the camera first; ~3 min; saved in marker_calibration.yaml.
#               Redo whenever a flag is moved. With flags, every touch corrects itself from the camera.
#   Red tint on the frame = beyond reach. After a click it checks reach/path/line before MOVE.
python3 gripper_markers.py --sheet gripper_markers.pdf   # regenerate the print sheet
