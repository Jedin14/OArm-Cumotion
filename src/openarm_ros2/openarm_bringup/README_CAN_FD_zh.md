# OpenArm（ROS 2）CAN-FD 开关说明（中文版）

本文档说明如何在启动时在经典 CAN 与 CAN‑FD 之间切换，以及为支持此能力所做的代码改动。

## 做了哪些改动

- 启动文件：新增 `can_fd` 启动参数，并贯通到 xacro，使其传递到硬件接口。
  - 文件：`openarm_bringup/launch/openarm.bimanual.launch.py`
    - 新增参数：`can_fd`（默认 `true`）
    - 通过 xacro `mappings` 传入键 `can_fd`

- Xacro：将 `can_fd` 在机器人描述中向下传递到 ros2_control 配置。
  - 文件：`openarm_description/urdf/robot/v10.urdf.xacro`
    - 增加 `xacro:arg name="can_fd" default="true"`
    - 将 `can_fd` 传给 `xacro:openarm_robot`
  - 文件：`openarm_description/urdf/robot/openarm_robot.xacro`
    - 在 `openarm_robot` 宏里新增 `can_fd` 参数
    - 将 `can_fd` 继续传入：
      - `openarm_description/urdf/ros2_control/openarm.bimanual.ros2_control.xacro`
      - `openarm_description/urdf/ros2_control/openarm.ros2_control.xacro`（单臂）

- 硬件：增加原始 `can_fd` 参数的日志，便于核对传参是否生效。
  - 文件：`openarm_ros2/openarm_hardware/src/v10_simple_hardware.cpp`
    - 新增日志：`Raw can_fd param: ...`，以及解析后的配置打印

## 如何使用

- 经典 CAN（禁用 CAN‑FD）：
```bash
ros2 launch openarm_bringup openarm.bimanual.launch.py can_fd:=false
```
- 启用 CAN‑FD（默认）：
```bash
ros2 launch openarm_bringup openarm.bimanual.launch.py can_fd:=true
```

CAN 口默认：
- 右臂：`right_can_interface:=can0`
- 左臂：`left_can_interface:=can1`

## 如何验证是否生效
查看 `ros2_control_node` 的日志中硬件插件打印：
- `Raw can_fd param: false`（或 `true`）
- 当传 `can_fd:=false` 时，`Configuration: ..., can_fd=disabled`
- 初始化时会看到：`Initializing OpenArm on canX with CAN-FD disabled...`

## 逐文件改动与代码片段

- `openarm_ros2/openarm_bringup/launch/openarm.bimanual.launch.py`
  - 新增参数与函数签名、xacro 映射：
```139:192:src/openarm_ros2/openarm_bringup/launch/openarm.bimanual.launch.py
        DeclareLaunchArgument(
            "can_fd",
            default_value="true",
            description="Enable CAN-FD for both arms (true) or use classic CAN (false).",
        ),
```
```39:66:src/openarm_ros2/openarm_bringup/launch/openarm.bimanual.launch.py
def generate_robot_description(context: LaunchContext, description_package, description_file,
                               arm_type, use_fake_hardware, can_fd, right_can_interface, left_can_interface):
    # ... existing code ...
    can_fd_str = context.perform_substitution(can_fd)
    # ... existing code ...
    robot_description = xacro.process_file(
        xacro_path,
        mappings={
            "arm_type": arm_type_str,
            "bimanual": "true",
            "use_fake_hardware": use_fake_hardware_str,
            "ros2_control": "true",
            "can_fd": can_fd_str,
            "right_can_interface": right_can_interface_str,
            "left_can_interface": left_can_interface_str,
        }
    ).toprettyxml(indent="  ")
```
```71:104:src/openarm_ros2/openarm_bringup/launch/openarm.bimanual.launch.py
def robot_nodes_spawner(context: LaunchContext, description_package, description_file,
                        arm_type, use_fake_hardware, controllers_file, can_fd, right_can_interface, left_can_interface, arm_prefix):
    # ... existing code ...
    robot_description = generate_robot_description(
        context, description_package, description_file, arm_type, use_fake_hardware, can_fd, right_can_interface, left_can_interface,
    )
```
```194:215:src/openarm_ros2/openarm_bringup/launch/openarm.bimanual.launch.py
    can_fd = LaunchConfiguration("can_fd")
    # ... existing code ...
    robot_nodes_spawner_func = OpaqueFunction(
        function=robot_nodes_spawner,
        args=[description_package, description_file, arm_type,
              use_fake_hardware, controllers_file, can_fd, rightcan_interface, left_can_interface, arm_prefix]
    )
```

- `openarm_description/urdf/robot/v10.urdf.xacro`
  - 新增参数并传入 `openarm_robot`：
```42:46:src/openarm_description/urdf/robot/v10.urdf.xacro
  <xacro:arg name="can_fd" default="true" />
```
```70:82:src/openarm_description/urdf/robot/v10.urdf.xacro
  right_arm_base_rpy="$(arg right_arm_base_rpy)"
  left_arm_base_rpy="$(arg left_arm_base_rpy)"
  can_fd="$(arg can_fd)"
  />
```

- `openarm_description/urdf/robot/openarm_robot.xacro`
  - 在宏签名声明 `can_fd`，并传入 ros2_control 宏：
```31:40:src/openarm_description/urdf/robot/openarm_robot.xacro
            right_arm_prefix:='right_'
            can_fd:=true
            "
            >
```
```100:111:src/openarm_description/urdf/robot/openarm_robot.xacro
            left_arm_prefix="${left_arm_prefix}"
            right_arm_prefix="${right_arm_prefix}"
            hand="${hand}"
            can_fd="${can_fd}"/>
```
```160:169:src/openarm_description/urdf/robot/openarm_robot.xacro
        <xacro:openarm_arm_ros2_control
            arm_type="${arm_type}"
            arm_prefix="${arm_prefix_modified}"
            can_interface="${can_interface}"
            use_fake_hardware="${use_fake_hardware}"
            fake_sensor_commands="${fake_sensor_commands}"
            hand="${hand}"
            bimanual="false"
            can_fd="${can_fd}"/>
```

- `openarm_ros2/openarm_hardware/src/v10_simple_hardware.cpp`
  - 打印原始参数与解析结果：
```52:66:src/openarm_ros2/openarm_hardware/src/v10_simple_hardware.cpp
  it = info.hardware_parameters.find("can_fd");
  std::string raw_can_fd = (it != info.hardware_parameters.end()) ? it->second : std::string("<unset>");
  if (it == info.hardware_parameters.end()) {
    can_fd_ = true;  // Default to true for V10
  } else {
    std::string value = it->second;
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    can_fd_ = (value == "true");
  }

  RCLCPP_INFO(rclcpp::get_logger("OpenArm_v10HW"),
              "Raw can_fd param: %s", raw_can_fd.c_str());
```
```63:66:src/openarm_ros2/openarm_hardware/src/v10_simple_hardware.cpp
  RCLCPP_INFO(rclcpp::get_logger("OpenArm_v10HW"),
              "Configuration: CAN=%s, arm_prefix=%s, hand=%s, can_fd=%s",
              can_interface_.c_str(), arm_prefix_.c_str(),
              hand_ ? "enabled" : "disabled", can_fd_ ? "enabled" : "disabled");
```
```120:126:src/openarm_ros2/openarm_hardware/src/v10_simple_hardware.cpp
  RCLCPP_INFO(rclcpp::get_logger("OpenArm_v10HW"),
              "Initializing OpenArm on %s with CAN-FD %s...",
              can_interface_.c_str(), can_fd_ ? "enabled" : "disabled");
  openarm_ =
      std::make_unique<openarm::can::socket::OpenArm>(can_interface_, can_fd_);
``` 