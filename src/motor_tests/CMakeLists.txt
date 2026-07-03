cmake_minimum_required(VERSION 3.16)
project(controll_test)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# 包含头文件目录
include_directories(include)

# 添加源文件
set(SOURCES
    src/canbus.cpp
    src/motor.cpp
    src/motor_control.cpp
)

# 删除原始版本可执行文件（pos_vel_test）

# 将带数据读取版本重命名为 test_single_motor
add_executable(test_single_motor ${SOURCES} pos_vel_test_with_data.cpp)

# 将增强版重命名为 test_motors_onebyone
add_executable(test_motors_onebyone ${SOURCES} pos_vel_test_enhanced.cpp)

# 将批量控制版本重命名为 test_motors_simulatious
add_executable(test_motors_simulatious ${SOURCES} pos_vel_test_enhanced_batch.cpp)

# 新增：本地 motor_check_all 功能（使用本目录内 hpp/cpp）
add_executable(motor_check_all_local ${SOURCES} motor_check_all_local.cpp)

# 新增：本地模式设置工具
add_executable(motor_mode_set_local ${SOURCES} motor_mode_set_local.cpp)

# 链接库
target_link_libraries(test_single_motor 
    pthread
)

target_link_libraries(test_motors_onebyone 
    pthread
)

target_link_libraries(test_motors_simulatious 
    pthread
)

target_link_libraries(motor_check_all_local 
    pthread
)

target_link_libraries(motor_mode_set_local 
    pthread
)

# 设置包含目录
target_include_directories(test_single_motor PRIVATE 
    include
)

target_include_directories(test_motors_onebyone PRIVATE 
    include
)

target_include_directories(test_motors_simulatious PRIVATE 
    include
)

target_include_directories(motor_check_all_local PRIVATE 
    include
)

target_include_directories(motor_mode_set_local PRIVATE 
    include
)
