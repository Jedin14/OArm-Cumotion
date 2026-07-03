#!/bin/bash

# 本地电机控制模式/在线状态检查脚本（基于 controll_test 本地实现）

echo "=== 本地电机检查 (motor_check_all_local) ==="
echo ""

# 需要 root 权限（访问 CAN 接口）
if [ "$EUID" -ne 0 ]; then
    echo "错误: 请使用 sudo 运行此脚本"
    echo "用法: sudo ./test_motor_check_all.sh"
    exit 1
fi

# 构建
if [ ! -d "build" ] || [ ! -f "build/motor_check_all_local" ]; then
    echo "未检测到可执行文件，正在构建..."
    ./build.sh || { echo "构建失败"; exit 1; }
fi

# 检查并启用 can0（必要时）
if ! ip link show can0 >/dev/null 2>&1; then
    echo "错误: can0 接口不存在，请先配置:"
    echo "  sudo ip link set can0 type can bitrate 1000000"
    echo "  sudo ip link set up can0"
    exit 1
fi
if ! ip link show can0 | grep -q "UP"; then
    echo "提示: 启用 can0..."
    ip link set up can0
fi

# 可选：如需要检查 can1，可取消以下注释
# if ip link show can1 >/dev/null 2>&1 && ! ip link show can1 | grep -q "UP"; then
#     echo "提示: 启用 can1..."
#     ip link set up can1
# fi

# 运行
./build/motor_check_all_local 