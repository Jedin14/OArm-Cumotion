#!/bin/bash

# OpenArm Bimanual MoveIt Demo with CAN 2.0 Setup Script
# This script configures CAN interfaces and launches the MoveIt demo

# Source ROS2 environment
echo "Sourcing ROS2 environment..."
cd ~/openarmx_ws/src/openarm_ros2/openarm_bimanual_moveit_config/
source ../../../install/local_setup.bash

echo "Setting up CAN interfaces for OpenArm Bimanual MoveIt Demo..."

# Configure CAN0 interface
echo "Configuring CAN0 interface..."
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up

if [ $? -eq 0 ]; then
    echo "CAN0 interface configured successfully"
else
    echo "Error: Failed to configure CAN0 interface"
    exit 1
fi

# Configure CAN1 interface
echo "Configuring CAN1 interface..."
sudo ip link set can1 down  
sudo ip link set can1 type can bitrate 1000000
sudo ip link set can1 up

if [ $? -eq 0 ]; then
    echo "CAN1 interface configured successfully"
else
    echo "Error: Failed to configure CAN1 interface"
    exit 1
fi

# Verify CAN interfaces are up
echo "Verifying CAN interfaces..."
ip link show can0 | grep -q "UP" && echo "CAN0 is UP" || echo "Warning: CAN0 may not be UP"
ip link show can1 | grep -q "UP" && echo "CAN1 is UP" || echo "Warning: CAN1 may not be UP"

echo ""
echo "CAN interfaces configured. Starting MoveIt demo with CAN 2.0 (non-FD)..."
echo "Launch command: ros2 launch openarm_bimanual_moveit_config demo.launch.py can_fd:=false"
echo ""

# Launch MoveIt demo with CAN 2.0 configuration
ros2 launch openarm_bimanual_moveit_config demo.launch.py can_fd:=false
