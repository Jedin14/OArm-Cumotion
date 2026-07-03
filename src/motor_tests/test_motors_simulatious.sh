#!/bin/bash

echo "=== 批量同时电机测试启动脚本 (原批量控制) ==="
echo ""

# 检查可执行文件是否存在
if [ ! -f "build/test_motors_simulatious" ]; then
    echo "可执行文件不存在，开始编译..."
    ./build.sh
    if [ $? -ne 0 ]; then
        echo "编译失败！"
        exit 1
    fi
fi

echo "启动批量同时控制测试程序..."
echo "测试内容："
echo "  - 一次性控制所有16个电机"
echo "  - 同时运动到目标位置（0.5rad或-0.5rad）"
echo "  - 然后同时回零位（0.0rad）"
echo "  - 实时显示所有电机状态"
echo ""

# 运行批量同时控制测试
./build/test_motors_simulatious

echo ""
echo "测试完成！"
