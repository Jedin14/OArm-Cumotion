#!/bin/bash

echo "=== 逐个电机测试启动脚本 (原增强版) ==="
echo ""

# 检查可执行文件是否存在
if [ ! -f "build/test_motors_onebyone" ]; then
    echo "可执行文件不存在，开始编译..."
    ./build.sh
    if [ $? -ne 0 ]; then
        echo "编译失败！"
        exit 1
    fi
fi

echo "启动逐个电机测试程序..."
echo "测试内容："
echo "  - can0: Right Arm Joint 1-7 + Gripper"
echo "  - can1: Left Arm Joint 1-7 + Gripper"
echo "  - 每个电机转动0-0.5rad"
echo "  - 实时显示位置/速度/扭矩数据"
echo ""

# 运行逐个测试
./build/test_motors_onebyone

echo ""
echo "测试完成！"
