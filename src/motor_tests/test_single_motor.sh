#!/bin/bash

# 单电机测试运行脚本（原：位置速度控制测试 - 带数据读取功能）

echo "=== 单电机测试 (原带数据读取版本) ==="
echo ""

# 检查是否以root权限运行
if [ "$EUID" -ne 0 ]; then
    echo "错误: 请使用sudo权限运行此脚本"
    echo "使用方法: sudo ./test_single_motor.sh"
    exit 1
fi

# 检查can0接口是否存在
if ! ip link show can0 >/dev/null 2>&1; then
    echo "错误: can0接口不存在"
    echo "请先配置CAN接口:"
    echo "  sudo ip link set can0 type can bitrate 1000000"
    echo "  sudo ip link set up can0"
    exit 1
fi

# 检查can0接口是否已启用
if ! ip link show can0 | grep -q "UP"; then
    echo "警告: can0接口未启用，正在启用..."
    ip link set up can0
fi

echo "CAN接口状态:"
ip link show can0
echo ""

# 检查可执行文件是否存在
if [ ! -f "build/test_single_motor" ]; then
    echo "错误: 可执行文件不存在，请先运行 ./build.sh"
    exit 1
fi

echo "开始运行单电机测试..."
echo "测试参数:"
echo "  - 电机ID: 001 (SlaveID=1, MasterID=0x11)"
echo "  - 电机类型: DM4310"
echo "  - 目标位置: 0.5 rad"
echo "  - 目标速度: 0.2 rad/s"
echo "  - 数据读取: 实时显示位置/速度/扭矩数据"
echo ""

# 运行测试
cd build
./test_single_motor

echo ""
echo "测试完成!"
