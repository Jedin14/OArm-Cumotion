# 位置速度控制模式测试

用于测试电机的位置速度控制（POS_VEL）功能，支持实时数据读取与多电机测试。

## 功能特性

- ✅ 单电机测试（实时数据读取）
- ✅ 逐个电机测试（can0+can1，16台）
- ✅ 同时批量测试（16台同时运动/回零）
- ✅ 实时读取电机位置、速度、扭矩数据
- ✅ 完整的电机启用/禁用流程
- ✅ 基于 openarm_hardware API

## 项目结构

```
motor_tests/
├── CMakeLists.txt
├── build.sh
├── test_single_motor.sh            # 单电机测试
├── test_motors_onebyone.sh         # 逐个电机测试
├── test_motors_simulatious.sh      # 同时批量测试
├── test_motors_controll_mode.sh    # 模式/在线状态检查
├── test_motors_set_and_check.sh    # 一键设置并检查
├── pos_vel_test_with_data.cpp
├── pos_vel_test_enhanced.cpp
├── pos_vel_test_enhanced_batch.cpp
├── include/openarm_hardware/
│   ├── canbus.hpp
│   ├── motor.hpp
│   ├── motor_control.hpp
│   └── openarm_hardware.hpp
├── src/
│   ├── canbus.cpp
│   ├── motor.cpp
│   ├── motor_control.cpp
└── build/
```

## 快速开始

### 1. 构建
```bash
cd motor_tests
./build.sh
```

### 2. 运行

- 单电机测试（推荐，带实时数据）：
```bash
sudo ./test_single_motor.sh
```

- 逐个电机测试（依次测试 16 台电机）：
```bash
sudo ./test_motors_onebyone.sh
```

- 同时批量测试（16 台电机同时运动与回零）：
```bash
sudo ./test_motors_simulatious.sh
```

- 电机控制模式/在线状态检查：
```bash
sudo ./test_motors_controll_mode.sh
```

- 一键设置并检查控制模式：
```bash
# 把 can0 的 1,2,3 号电机切到 MIT，并保存
sudo ./test_motors_set_and_check.sh -m MIT --ids 1,2,3 --save

# 把 can0 的所有电机切到 POS_VEL（不保存）
sudo ./test_motors_set_and_check.sh -m POS_VEL -a

# 同时把 can0 和 can1 的 1,2,3 号切到 VEL（不保存）
sudo ./test_motors_set_and_check.sh -m VEL --ids 1,2,3 --both
```

## 参数说明（test_motors_set_and_check.sh）
- `-m, --mode`：设置的控制模式。取值：`MIT`、`POS_VEL`、`VEL`、`POS_FORCE`（也可用 1/2/3/4）。必填
- `-i, --interface`：指定 CAN 接口，默认 `can0`。示例：`-i can1`
- `--ids`：指定电机 ID 列表（1-8），逗号分隔。示例：`--ids 1,2,3`
- `-a, --all`：该接口上全部电机（1-8）。与 `--ids` 二选一
- `--save`：写入电机内部存储，掉电保存（执行后建议断电重启验证）
- `--both`：同时在 `can0` 与 `can1` 执行设置（依次设置两个接口后再统一检查）

## 测试参数（默认）

| 参数 | 值 | 说明 |
|------|-----|------|
| 电机ID | 001 | SlaveID=1, MasterID=0x11 |
| 电机类型 | DM4310 | 达妙电机型号 |
| 目标位置 | 0.5 rad | 单电机/逐个/批量测试默认值 |
| 目标速度 | 0.2 rad/s | 单电机默认值（增强版本内部可能提高） |
| CAN接口 | can0 (+ can1) | 单电机默认 can0；逐个/批量会使用 can0+can1 |

## 输出示例（单电机）
```
=== 单电机测试 (原带数据读取版本) ===
CAN接口状态: ...
  循环    目标位置    实际位置    目标速度    实际速度    实际扭矩    位置误差
   1       0.500       0.047       0.200      -0.007      -0.007       0.453
  ...
```

## 故障排除

1) 权限问题
```bash
sudo ./test_single_motor.sh
```

2) CAN 接口未配置
```bash
sudo ip link set can0 type can bitrate 1000000
sudo ip link set up can0
ip link show can0
```

3) 电机无响应
- 检查接线与电机 ID
- 确认 can0/can1 状态
- 使用 `candump can0` 观察总线

4) 编译失败
```bash
sudo apt install build-essential cmake
./build.sh
```

## 依赖
- OS: Ubuntu 20.04+ / Linux
- CMake 3.16+，GCC 7.0+
- CAN 接口权限（需 sudo）
- 达妙 DM4310 电机

