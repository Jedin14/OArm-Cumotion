# old_version

Files that the running code no longer uses, kept for reference. Nothing here is
imported or run by the current stack, and moving anything out of this folder is
not needed for the robot, the GUI or the pick stack to work.

| Path | What it was |
| --- | --- |
| `auto_box.py`, `native/run_auto_box.sh` | The OpenCV pick tool for the old YOLO box model. The model itself (`box_target_detector/`) was deleted in commit 25e81ff and is in git history before it |
| `README_2026-10-13.md` | The previous README: container fallback, YOLO and VLM sections, older troubleshooting |
| `install_with_vlm.sh`, `check_isolation_with_vlm.sh`, `INSTALL_with_vlm.md`, `gitignore_with_old_models` | The installer, isolation check, install guide and `.gitignore` as they were with the old VLM (PaliGemma) detector and graspnet |
| `container/` | The container-era launcher (`run_launch_everything.sh`, which sourced the root `install/`) and that container build's `build/` and `install/` (July 2026, git-ignored). The native environment (`native/`) replaced it |
| `openarm_spheres.yml` | `native/make_collision_spheres.py`'s output, already merged into `openarm.yml` |
| `native/openarm_native.yml` | An early copy (Aug 2026) of the cuMotion robot model, replaced by `openarm.yml` |
| `src_backups/demo.launch.py.backup` | The MoveIt demo launch as it was in July, before the single-spawner fix (from `src/openarm_ros2/openarm_bimanual_moveit_config/launch/`) |
| `src_backups/ros2_lelab_bridge.py.pre-v21-sync.bak` | A teleop bridge backup (from `src/qnbot_teleoperator/scripts/`) |
| `calibration_backups/` | Arm models, anchors and a depth correction from before they were replaced or rejected. New backups are written here too: the box calibration (`arm_model.before_boxcal_*.yaml`) and the tool weighing (`*.before_weigh_*`) |
| `box_picker_2026-11-03/` | The box picker's files as they were on 3 Nov, replaced on 4 Nov by the version from `box_picker_for_robot.zip` (the accelerometer over ROS, the flat-box rest rule) |
| `armcal_samples/` | Calibration recordings superseded by newer ones (the fits use only the newest of each kind, in `calibration/armcal_samples/`) |
