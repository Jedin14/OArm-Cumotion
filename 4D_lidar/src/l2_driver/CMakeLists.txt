cmake_minimum_required(VERSION 3.10)
project(l2_driver)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(Threads REQUIRED)

# Vendored Unitree SDK (closed-source static library, x86_64 and aarch64 builds).
set(L2_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/unilidar_sdk2")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(L2_SDK_ARCH aarch64)
else()
  set(L2_SDK_ARCH x86_64)
endif()
add_library(unilidar_sdk2 STATIC IMPORTED)
set_target_properties(unilidar_sdk2 PROPERTIES
  IMPORTED_LOCATION "${L2_SDK_DIR}/lib/${L2_SDK_ARCH}/libunilidar_sdk2.a")

add_executable(l2_driver_node src/l2_driver_node.cpp)
target_include_directories(l2_driver_node PRIVATE include)
# SYSTEM: do not apply our warning flags to the vendor headers
target_include_directories(l2_driver_node SYSTEM PRIVATE "${L2_SDK_DIR}/include")
target_link_libraries(l2_driver_node unilidar_sdk2 Threads::Threads)
ament_target_dependencies(l2_driver_node rclcpp sensor_msgs geometry_msgs diagnostic_msgs tf2_ros)

install(TARGETS l2_driver_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_timing test/test_timing.cpp)
  target_include_directories(test_timing PRIVATE include)
endif()

ament_package()
