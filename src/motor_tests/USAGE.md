# 快速使用指南

## 🚀 快速开始

### 1. 构建
```bash
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
# can0, 指定电机并保存
sudo ./test_motors_set_and_check.sh -m MIT --ids 1,2,3 --save
# can0, 全部电机（不保存）
sudo ./test_motors_set_and_check.sh -m POS_VEL -a
# can0+can1, 指定电机（不保存）
sudo ./test_motors_set_and_check.sh -m VEL --ids 1,2,3 --both
```

## ⚙️ 参数说明（test_motors_set_and_check.sh）
- `-m, --mode`：设置的控制模式。取值：`MIT`、`POS_VEL`、`VEL`、`POS_FORCE`（也可用 1/2/3/4）。必填
- `-i, --interface`：指定 CAN 接口，默认 `can0`。示例：`-i can1`
- `--ids`：指定电机 ID 列表（1-8），逗号分隔。示例：`--ids 1,2,3`
- `-a, --all`：该接口上全部电机（1-8）。与 `--ids` 二选一
- `--save`：写入电机内部存储，掉电保存（执行后建议断电重启验证）
- `--both`：同时在 `can0` 与 `can1` 执行设置（依次设置两个接口后再统一检查）

## ⚠️ 注意事项
1. 需要 sudo 权限运行（访问 CAN 接口）
2. 确保 can0（以及 can1）已配置并 UP
3. 确保电机连接与 ID 配置正确

## 🛠️ 故障排除
- 配置 CAN 接口：
```bash
sudo ip link set can0 type can bitrate 1000000
sudo ip link set up can0
ip link show can0
```

- 重新构建：
```bash
./build.sh
```
