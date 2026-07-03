#!/bin/bash

# 一键设置电机控制模式并检查结果
# 用法示例：
#   sudo ./test_motors_set_and_check.sh -m MIT --ids 1,2,3 --save
#   sudo ./test_motors_set_and_check.sh -m POS_VEL -a
#   sudo ./test_motors_set_and_check.sh -m VEL -i can1 --ids 4,5

set -e

if [ "$EUID" -ne 0 ]; then
  echo "错误: 请使用 sudo 运行此脚本"
  exit 1
fi

MODE=""
IFACE="can0"
IDS=""
ALL=false
SAVE=false
BOTH=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--mode)
      MODE="$2"; shift 2;;
    -i|--interface)
      IFACE="$2"; shift 2;;
    --ids)
      IDS="$2"; shift 2;;
    -a|--all)
      ALL=true; shift;;
    --save)
      SAVE=true; shift;;
    --both)
      BOTH=true; shift;;
    -h|--help)
      echo "用法: sudo $0 -m MIT|POS_VEL|VEL|POS_FORCE [-i canX|--both] [--ids 1,2,3 | -a] [--save]"; exit 0;;
    *) echo "未知参数: $1"; exit 1;;
  esac
done

if [ -z "$MODE" ]; then
  echo "错误: 必须指定 -m/--mode"; exit 1;
fi

# 构建（如缺失）
if [ ! -d "build" ] || [ ! -f "build/motor_mode_set_local" ] || [ ! -f "build/motor_check_all_local" ]; then
  ./build.sh
fi

# 确定接口列表
IFACES=()
if $BOTH; then
  IFACES=(can0 can1)
else
  IFACES=($IFACE)
fi

# 检查/启用各接口
for IFX in "${IFACES[@]}"; do
  if ! ip link show "$IFX" >/dev/null 2>&1; then
    echo "错误: 接口 $IFX 不存在，请先配置"; exit 1;
  fi
  if ! ip link show "$IFX" | grep -q "UP"; then
    echo "提示: 启用 $IFX..."; ip link set up "$IFX";
  fi
done

# 组装 ids/all 选项
ID_OPTS=()
if $ALL; then
  ID_OPTS+=( -a )
elif [ -n "$IDS" ]; then
  ID_OPTS+=( --ids "$IDS" )
else
  echo "错误: 需提供 --ids 或 -a"; exit 1;
fi

# 设置循环
echo "=== 设置控制模式 ==="
for IFX in "${IFACES[@]}"; do
  CMD=("./build/motor_mode_set_local" -i "$IFX" -m "$MODE" "${ID_OPTS[@]}")
  if $SAVE; then CMD+=( --save ); fi
  echo "[接口 $IFX]"
  "${CMD[@]}"
  echo ""
  # 小等待
  sleep 0.3
done

# 检查
echo -e "\n=== 检查控制模式 ==="
./build/motor_check_all_local 