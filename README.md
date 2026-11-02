# OpenArm pick and place (beast)

A bimanual OpenArm (ROS 2 Humble) that picks boxes with a vacuum cup on the left arm:
MoveIt + cuMotion for motion, a RealSense D455 head camera, and the SAM 3.1 box
picker to choose the box and the face to grip. It is driven from a web page or
over ROS topics under `/beast_arm`.

## Quick start

```bash
# once per boot, on beast (needs sudo): the arms' CAN FD buses
sudo ip link set can0 type can bitrate 1000000 dbitrate 5000000 fd on && sudo ip link set can0 up
sudo ip link set can1 type can bitrate 1000000 dbitrate 5000000 fd on && sudo ip link set can1 up

native/run_launch_everything.sh     # terminal 1: robot, MoveIt, cuMotion, camera; arms boot to navigation
native/run_web_gui.sh               # terminal 2: the web GUI + /beast_arm interface -> http://beast:8080
```

In the page: **Detect box**, then **Move**. Or from any ROS 2 machine on the network:

```bash
ros2 topic pub --once /beast_arm/arm_action std_msgs/msg/String "{data: ready_to_pick}"
ros2 topic echo /beast_arm/arm_action/status
```

## Documentation

| File | What |
| --- | --- |
| [docs/detailed_run.md](docs/detailed_run.md) | The whole run, start to finish: every step, every case, and what is published and received |
| [docs/cmds.md](docs/cmds.md) | Everyday commands |
| [docs/ros_topic.md](docs/ros_topic.md) | The `/beast_arm` ROS interface: topics, services, messages, options |
| [docs/INSTALL.md](docs/INSTALL.md) | Installing on a new machine |
| [native/README.md](native/README.md) | The native (no container) build environment |
| [box_picker/](box_picker/RUN_LIVE.md) | The SAM 3.1 box picker |

## Layout

| Path | What |
| --- | --- |
| `app/` | The pick side (Python), run from here by the scripts in `native/` |
| `app/web_gui.py` | The web GUI (port 8080) and the `/beast_arm` ROS interface |
| `app/pick_stack.py` | The same ROS interface without the page (`native/run_pick_stack.sh`) |
| `app/click_to_move.py` | The core app: options, motion, the touch; also the OpenCV click tool (`native/run_click_to_move.sh`) |
| `app/calibrate.py` | The OpenCV calibration window (`native/run_calibrate.sh`); its refit is used by the GUI |
| `app/box_model_node.py`, `app/box_picker_server.py` | The box model's ROS node and the picker process it runs |
| `app/robot_camera_calibration.py` | Camera pose from the arms (the GUI's camera Calibrate / Apply) |
| `app/ctm/` | The pick stack's modules: robot interface, touch, calibrations, box matching, ... (`app/ctm/common.py` defines every folder below) |
| `app/web/`, `app/launch/` | The page (HTML, JS, CSS); `web_gui.launch.py`, `pick_stack.launch.py` |
| `robot/` | The robot stack (terminal 1): `launch_everything.launch.py`, `boot_pre_pick.py`, `octomap_gater.py`; the robot model `openarm.yml` (cuMotion) and `openarm.urdf`; the navigation / pre_pick / drop poses `pick_place_states_{left,right}.yaml` and `record_states.py`, which records them |
| `calibration/` | What the calibrations and the GUI write and the picks read: `arm_model.yaml`, `armcal_anchors.yaml`, `touch_corrections.yaml`, `payload_left.yaml`, `camera_calibration.yaml`, `depth_correction.yaml` (when saved), `tools.yaml` (the tool past each gripper's fingertips), `cam_org.txt` (the head camera's mount history), and the raw recordings in `armcal_samples/` |
| `docs/` | The documents above; `docs/sheets/` the printable marker sheets |
| `native/` | Run scripts, the native build environment, `install.sh` and `check_isolation.sh` |
| `src/` | ROS packages: OpenArm description and drivers, cuMotion (with the patched planner), RealSense, `pick_interfaces` |
| `box_picker/` | The SAM 3.1 box picker (own venv: `box_picker/src/setup.sh`) |
| `old_version/` | Superseded files, kept: see [old_version/README.md](old_version/README.md) |
| `log/` | colcon's build logs (`native/build_ws.sh`) |

The modules under `ctm/` run from `app/`: `cd app && python3 -m ctm.board --live`.
