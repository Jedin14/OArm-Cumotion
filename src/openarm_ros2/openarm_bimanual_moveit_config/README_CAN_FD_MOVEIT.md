# MoveIt Demo CAN-FD 支持修改说明

本文档详细说明为 `openarm_bimanual_moveit_config/launch/demo.launch.py` 添加 `can_fd` 参数支持所做的修改。

## 修改背景

原始的 MoveIt demo 启动文件缺少 `can_fd` 参数支持，无法在启动时选择经典 CAN 或 CAN-FD 模式。为了与 `openarm.bimanual.launch.py` 保持一致，需要添加相同的 `can_fd` 参数支持。

## 修改的文件

- `src/openarm_ros2/openarm_bimanual_moveit_config/launch/demo.launch.py`

## 具体修改内容

### 1. 函数签名修改

**修改前：**
```python
def generate_robot_description(
    context: LaunchContext,
    description_package,
    description_file,
    arm_type,
    use_fake_hardware,
    right_can_interface,
    left_can_interface,
    arm_prefix,
):
```

**修改后：**
```python
def generate_robot_description(
    context: LaunchContext,
    description_package,
    description_file,
    arm_type,
    use_fake_hardware,
    can_fd,  # 新增参数
    right_can_interface,
    left_can_interface,
    arm_prefix,
):
```

### 2. 参数解析添加

**新增代码：**
```python
can_fd_str = context.perform_substitution(can_fd)
```

### 3. Xacro mappings 添加 can_fd

**修改前：**
```python
robot_description = xacro.process_file(
    xacro_path,
    mappings={
        "arm_type": arm_type_str,
        "bimanual": "true",
        "use_fake_hardware": use_fake_hardware_str,
        "ros2_control": "true",
        "left_can_interface": left_can_interface_str,
        "right_can_interface": right_can_interface_str,
        # arm_prefix unused inside xacro but kept for completeness
    },
).toprettyxml(indent="  ")
```

**修改后：**
```python
robot_description = xacro.process_file(
    xacro_path,
    mappings={
        "arm_type": arm_type_str,
        "bimanual": "true",
        "use_fake_hardware": use_fake_hardware_str,
        "ros2_control": "true",
        "can_fd": can_fd_str,  # 新增映射
        "left_can_interface": left_can_interface_str,
        "right_can_interface": right_can_interface_str,
        # arm_prefix unused inside xacro but kept for completeness
    },
).toprettyxml(indent="  ")
```

### 4. robot_nodes_spawner 函数签名修改

**修改前：**
```python
def robot_nodes_spawner(
    context: LaunchContext,
    description_package,
    description_file,
    arm_type,
    use_fake_hardware,
    controllers_file,
    right_can_interface,
    left_can_interface,
    arm_prefix,
):
```

**修改后：**
```python
def robot_nodes_spawner(
    context: LaunchContext,
    description_package,
    description_file,
    arm_type,
    use_fake_hardware,
    controllers_file,
    can_fd,  # 新增参数
    right_can_interface,
    left_can_interface,
    arm_prefix,
):
```

### 5. robot_nodes_spawner 函数调用修改

**修改前：**
```python
robot_description = generate_robot_description(
    context,
    description_package,
    description_file,
    arm_type,
    use_fake_hardware,
    right_can_interface,
    left_can_interface,
    arm_prefix,
)
```

**修改后：**
```python
robot_description = generate_robot_description(
    context,
    description_package,
    description_file,
    arm_type,
    use_fake_hardware,
    can_fd,  # 新增参数传递
    right_can_interface,
    left_can_interface,
    arm_prefix,
)
```

### 6. 启动参数声明添加

**新增代码：**
```python
DeclareLaunchArgument(
    "can_fd",
    default_value="true",
    description="Enable CAN-FD for both arms (true) or use classic CAN (false).",
),
```

### 7. LaunchConfiguration 添加

**新增代码：**
```python
can_fd = LaunchConfiguration("can_fd")
```

### 8. OpaqueFunction 参数传递修改

**修改前：**
```python
robot_nodes_spawner_func = OpaqueFunction(
    function=robot_nodes_spawner,
    args=[
        description_package,
        description_file,
        arm_type,
        use_fake_hardware,
        controllers_file,
        right_can_interface,
        left_can_interface,
        arm_prefix,
    ],
)
```

**修改后：**
```python
robot_nodes_spawner_func = OpaqueFunction(
    function=robot_nodes_spawner,
    args=[
        description_package,
        description_file,
        arm_type,
        use_fake_hardware,
        controllers_file,
        can_fd,  # 新增参数
        right_can_interface,
        left_can_interface,
        arm_prefix,
    ],
)
```

## 使用方法

### 经典 CAN（禁用 CAN-FD）
```bash
ros2 launch openarm_bimanual_moveit_config demo.launch.py can_fd:=false
```

### CAN-FD（默认启用）
```bash
ros2 launch openarm_bimanual_moveit_config demo.launch.py can_fd:=true
```

## 验证方法

启动后查看 `ros2_control_node` 日志，应该能看到：
- `Raw can_fd param: false`（当使用 `can_fd:=false` 时）
- `Configuration: ..., can_fd=disabled`（当禁用 CAN-FD 时）
- `Initializing OpenArm on canX with CAN-FD disabled...`（当禁用 CAN-FD 时）

## 构建要求

修改后需要重新构建包：
```bash
colcon build --packages-select openarm_bimanual_moveit_config
```

## 备份文件

原始文件已备份为：`demo.launch.py.backup`

## 注意事项

- 此修改与 `openarm.bimanual.launch.py` 的 `can_fd` 支持保持一致
- 默认值保持 `true`（启用 CAN-FD），与 V10 硬件默认配置一致
- 修改不影响 MoveIt 的其他功能，仅添加了 CAN 模式选择能力
