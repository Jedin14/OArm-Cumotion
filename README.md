# OpenArm IK Docker Environment

This repository contains the complete Dockerized environment for running the OpenArm bimanual exoskeleton with GPU-accelerated cuMotion kinematics, MoveIt 2, and real hardware interfaces.

## 🚀 Quick Start

### 1. Enter the Docker Environment
To launch the Isaac ROS development container, run the provided script:
```bash
cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common
./scripts/run_dev.sh
```

### 2. Launch the Robot and Planners (GPU + MoveIt)
Once inside the Docker container, source the workspace and start the main launch file:
```bash
source install/setup.bash
ros2 launch /workspaces/isaac_ros-dev/launch_everything.launch.py
```

### 3. Editing the Code in VS Code
To open the entire workspace in Visual Studio Code from your host machine, simply run:
```bash
code ~/workspaces/isaac_ros-dev
```

---

## 🛠️ Important Notes & Fixes Applied

- **cuMotion Planner (GPU IK)**: cuMotion is set as the default MoveIt planner for the 7-DOF arms (`left_joint_trajectory_controller`, `right_joint_trajectory_controller`).
- **Gripper Planning**: The cuMotion planner **does not support 1-DOF grippers**. 
  - If you attempt to plan for the grippers using cuMotion, the planner will safely reject the request to prevent crashes.
  - **Always switch the Planning Pipeline to `ompl` in RViz** when planning motions for `left_gripper` or `right_gripper`!
- **Gripper PID Tuning**: The DM4310 gripper motor gains have been carefully tuned (`Kp = 20.0`, `Kd = 0.5`) to eliminate high-frequency noise and vibrations while providing enough torque for accurate movement.
- **Core Dump Cleanup**: The `./scripts/run_dev.sh` startup script has been modified to automatically delete large `core.*` crash dump files from the workspace before the container starts, ensuring your hard drive does not run out of space.


sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000 dbitrate 5000000 fd on
sudo ip link set can0 up

sudo ip link set can1 down
sudo ip link set can1 type can bitrate 1000000 dbitrate 5000000 fd on
sudo ip link set can1 up