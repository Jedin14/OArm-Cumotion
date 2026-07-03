# openarm_utils 2.0

简介
----
openarm_utils 包提供两个核心功能：
1. 录制：从 /joint_states 按固定频率采样所有关节（双臂 + 夹爪）并生成 YAML 轨迹。
2. 回放：读取该 YAML，通过 FollowJointTrajectory 与 GripperCommand Action 控制器执行，支持多控制器并行与基础关节过滤。

快速开始
--------
编译：
```
colcon build --packages-select openarm_utils
```

启动机器人moveit包：
```
真机：
ros2 launch openarm_bimanual_moveit_config demo.launch.py
仿真：
ros2 launch openarm_bimanual_moveit_config demo.launch.py use_fake_hardware:=true
```


录制轨迹（默认 10Hz，手动开始/暂停）：
```
ros2 run openarm_utils record_joint_states_always --rate 20
```
按键：SPACE/p 开始/暂停；c 清空；w 保存退出；q 不保存退出。

生成的文件示例：joint_states_stream_YYYYMMDD_HHMMSS.yaml

回放全部关节（双臂 + 双夹爪）：
```
ros2 run openarm_utils play_joint_trajectory joint_states_stream_YYYYMMDD_HHMMSS.yaml --all-joints
```
只回放左臂：
```
ros2 run openarm_utils play_joint_trajectory <yaml> --left-arm
```
只测试左夹爪 Action：
```
ros2 run openarm_utils play_joint_trajectory <yaml> --action /left_gripper_controller/gripper_cmd
```

YAML 格式
---------
joint_names: [关节1, 关节2, ...]
points:
  - positions: [按 joint_names 顺序的浮点值]
    time_from_start: 相对录制起点的秒(固定步长推导)

脚本说明
--------
record_joint_states_always.py:
  - 按 --rate 频率采样 /joint_states。
  - 保存时使用递增 (i+1)*dt 生成 time_from_start。
play_joint_trajectory.py:
  - 将 YAML 中的轨迹转换为 FollowJointTrajectory.Goal 或一系列 GripperCommand.Goal。
  - 支持关节筛选(--joints / --left-arm / --right-arm / --both-arms / --all-joints)。
  - 同步执行各控制器（当前版本对夹爪使用简单时间调度，不做反馈驱动）。

主要参数
--------
录制：
--rate Hz       采样频率
--topic 话题     默认 /joint_states
--outfile 路径   自定义输出文件名

回放：
--rate-scale f    时间缩放（>1 加速，<1 减速）
--joints 列表     指定一组关节名称
--left-arm        仅左臂关节
--right-arm       仅右臂关节
--both-arms       双臂不含夹爪
--all-joints      全部关节（默认）
--action 名称     指定单一 Action（例如某个 gripper_cmd）

典型流程
--------
1. 启动机器人及控制器。
2. 使用 record_joint_states_always 录制：进入运动 → SPACE 开始 → SPACE 暂停 → w 保存。
3. 用 play_joint_trajectory 回放：先 --all-joints 验证，再细化到某侧手臂或夹爪。
4. 若关节名称对不上（警告提示未找到），确认 YAML 中 joint_names 顺序和实际控制器关节命名一致。

注意事项
--------
1. 该版本的 time_from_start 为固定间隔推导，不是真实反馈时间；复杂加减速或控制器启动延迟可能造成轻微不同步。
2. 录制频率过低会导致轨迹点稀疏；过高则文件增大但对夹爪的连续变化意义不大。
3. GripperCommand Action 多次发送相近目标意义有限，建议录制前确保夹爪动作有明确开合幅度。

常见问题与排查
----------------
“Warning: Joint 'xxx' not found”: YAML 中无该关节名称 → 检查命名或录制阶段是否已加载对应控制器。
回放看不到夹爪动作：录制时夹爪未变化或变化幅度很小（数据几乎恒定）。
轨迹执行失败：控制器 Action 名称与实际运行的控制器不匹配（检查启动文件路径与 action 名）。
播放速度不对：使用 --rate-scale 调整或提高录制频率。

非商业开源许可
----------------
本项目仅授权用于非商业的学习、教学、学术研究及开源社区协作。任何商业用途（含内部评估转产品化、作为解决方案一部分、收费服务等）须事先获得“长数机器人有限公司”书面授权。

授权主体信息：
- 公司：长数机器人有限公司
- 电话：+86 17746530375
- 邮箱：openarmrobot@gmail.com
- 地址：天津经济技术开发区西区新业八街

许可要点：
1. 可自由查看、复制、修改、分发（限非商业用途），需保留本许可段。
2. 禁止去除或修改版权/许可声明；禁止将本项目嵌入任何商业产品/服务后再分发。
3. 商用授权申请：邮件说明预期用途、产品形态、时间计划，待公司评估后签署授权协议。
4. 本软件按“现状”提供，不承担因使用造成的任何直接或间接损失责任。
5. 违反条款即自动终止使用权，公司保留追究法律责任的权利。

免责声明：
本项目不提供适销性或特定用途适用性保证；使用者需自行评估风险与安全性。

版本：License v1.0 (2025-11-17)

