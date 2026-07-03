// Copyright 2025 OpenArm Control Test
//
// 升级版批量位置速度控制模式测试程序
// 一次性控制所有电机运动到位，然后同时回零位
// 支持多种速度测试

#include <iostream>
#include <chrono>
#include <thread>
#include <unistd.h>
#include <iomanip>
#include <vector>
#include <string>
#include <memory>

// 包含openarm_hardware的头文件
#include "openarm_hardware/canbus.hpp"
#include "openarm_hardware/motor.hpp"
#include "openarm_hardware/motor_control.hpp"

// 电机配置结构
struct MotorConfig {
    std::string can_interface;
    int motor_id;
    DM_Motor_Type motor_type;
    uint16_t slave_id;
    uint16_t master_id;
    std::string description;
    bool reverse_direction;  // 是否反向旋转
};

// 电机控制结构
struct MotorController {
    std::unique_ptr<CANBus> canbus;
    std::unique_ptr<MotorControl> motor_control;
    std::unique_ptr<Motor> motor;
    MotorConfig config;
    bool is_enabled;
    double target_position;
    double current_position;
    double current_velocity;
    double current_torque;
};

// 电机配置数组 - 根据您的要求配置
const std::vector<MotorConfig> motor_configs = {
    // CAN0 (Right Arm) - Joint 1-7: 正向旋转, Gripper: 反向旋转
    {"can0", 1, DM_Motor_Type::DM4310, 0x01, 0x11, "Right Arm Joint 1", false},   // 正向
    {"can0", 2, DM_Motor_Type::DM4310, 0x02, 0x12, "Right Arm Joint 2", false},   // 正向
    {"can0", 3, DM_Motor_Type::DM4310, 0x03, 0x13, "Right Arm Joint 3", false},   // 正向
    {"can0", 4, DM_Motor_Type::DM4310, 0x04, 0x14, "Right Arm Joint 4", false},   // 正向
    {"can0", 5, DM_Motor_Type::DM4310, 0x05, 0x15, "Right Arm Joint 5", false},   // 正向
    {"can0", 6, DM_Motor_Type::DM4310, 0x06, 0x16, "Right Arm Joint 6", false},   // 正向
    {"can0", 7, DM_Motor_Type::DM4310, 0x07, 0x17, "Right Arm Joint 7", false},   // 正向
    {"can0", 8, DM_Motor_Type::DM4310, 0x08, 0x18, "Right Arm Gripper", true},    // 反向
    
    // CAN1 (Left Arm) - Joint 1-3: 反向, Joint 4: 正向, Joint 5-7: 反向, Gripper: 反向
    {"can1", 1, DM_Motor_Type::DM4310, 0x01, 0x11, "Left Arm Joint 1", false},     // 反向
    {"can1", 2, DM_Motor_Type::DM4310, 0x02, 0x12, "Left Arm Joint 2", false},     // 反向
    {"can1", 3, DM_Motor_Type::DM4310, 0x03, 0x13, "Left Arm Joint 3", false},     // 正向
    {"can1", 4, DM_Motor_Type::DM4310, 0x04, 0x14, "Left Arm Joint 4", false},    // 正向
    {"can1", 5, DM_Motor_Type::DM4310, 0x05, 0x15, "Left Arm Joint 5", true},     // 反向
    {"can1", 6, DM_Motor_Type::DM4310, 0x06, 0x16, "Left Arm Joint 6", true},     // 反向
    {"can1", 7, DM_Motor_Type::DM4310, 0x07, 0x17, "Left Arm Joint 7", true},     // 反向
    {"can1", 8, DM_Motor_Type::DM4310, 0x08, 0x18, "Left Arm Gripper", true}      // 反向
};

// 测试速度数组
const std::vector<double> test_velocities = {0.1, 0.3, 0.6, 0.9};

// 初始化电机控制器
bool init_motor_controller(MotorController& controller) {
    try {
        // 初始化CAN总线
        controller.canbus = std::make_unique<CANBus>(controller.config.can_interface, CAN_MODE_CLASSIC);
        
        // 初始化电机控制
        controller.motor_control = std::make_unique<MotorControl>(*controller.canbus);
        
        // 创建电机对象
        controller.motor = std::make_unique<Motor>(controller.config.motor_type, 
                                                   controller.config.slave_id, 
                                                   controller.config.master_id);
        
        // 将电机添加到控制器
        controller.motor_control->addMotor(*controller.motor);
        
        // 设置目标位置
        controller.target_position = controller.config.reverse_direction ? -0.5 : 0.5;
        
        // 启用电机
        std::cout << "启用电机: " << controller.config.description << std::endl;
        controller.motor_control->enable(*controller.motor);
        controller.is_enabled = true;
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "初始化电机失败 " << controller.config.description << ": " << e.what() << std::endl;
        controller.is_enabled = false;
        return false;
    }
}

// 更新电机状态
void update_motor_status(MotorController& controller) {
    if (controller.is_enabled && controller.motor) {
        controller.current_position = controller.motor->getPosition();
        controller.current_velocity = controller.motor->getVelocity();
        controller.current_torque = controller.motor->getTorque();
    }
}

// 批量控制所有电机
void batch_control_motors(std::vector<MotorController>& controllers, double target_position, double target_velocity) {
    for (auto& controller : controllers) {
        if (controller.is_enabled) {
            controller.motor_control->controlPosVel(*controller.motor, target_position, target_velocity);
        }
    }
}

// 显示电机状态表格
void display_motor_status(const std::vector<MotorController>& controllers, int cycle, const std::string& phase, double target_velocity) {
    std::cout << "\n=== " << phase << " - 循环 " << cycle << " (速度: " << target_velocity << " rad/s) ===" << std::endl;
    
    // 表头
    std::cout << std::setw(20) << "电机名称" 
              << std::setw(12) << "目标位置" 
              << std::setw(12) << "实际位置" 
              << std::setw(12) << "目标速度" 
              << std::setw(12) << "实际速度" 
              << std::setw(12) << "实际扭矩" 
              << std::setw(12) << "位置误差" << std::endl;
    std::cout << std::setw(20) << "--------" 
              << std::setw(12) << "--------" 
              << std::setw(12) << "--------" 
              << std::setw(12) << "--------" 
              << std::setw(12) << "--------" 
              << std::setw(12) << "--------" 
              << std::setw(12) << "--------" << std::endl;
    
    // 显示每个电机的状态
    for (const auto& controller : controllers) {
        if (controller.is_enabled) {
            double position_error = controller.target_position - controller.current_position;
            std::cout << std::setw(20) << controller.config.description
                      << std::setw(12) << std::fixed << std::setprecision(3) << controller.target_position
                      << std::setw(12) << std::fixed << std::setprecision(3) << controller.current_position
                      << std::setw(12) << std::fixed << std::setprecision(3) << target_velocity
                      << std::setw(12) << std::fixed << std::setprecision(3) << controller.current_velocity
                      << std::setw(12) << std::fixed << std::setprecision(3) << controller.current_torque
                      << std::setw(12) << std::fixed << std::setprecision(3) << position_error
                      << std::endl;
        }
    }
}

// 批量运动到目标位置
bool batch_move_to_target(std::vector<MotorController>& controllers, double target_position, double target_velocity, const std::string& phase_name) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "=== " << phase_name << " ===" << std::endl;
    std::cout << "目标位置: " << target_position << " rad" << std::endl;
    std::cout << "目标速度: " << target_velocity << " rad/s" << std::endl;
    
    int max_cycles = 200;  // 增加最大控制循环数，适应低速测试
    int display_interval = 20;  // 每20次循环显示一次状态
    
    for (int cycle = 1; cycle <= max_cycles; cycle++) {
        // 批量发送控制命令
        batch_control_motors(controllers, target_position, target_velocity);
        
        // 更新所有电机状态
        for (auto& controller : controllers) {
            update_motor_status(controller);
        }
        
        // 定期显示状态
        if (cycle % display_interval == 0) {
            display_motor_status(controllers, cycle, phase_name, target_velocity);
        }
        
        // 检查是否所有电机都到达目标位置（容差0.05rad）
        bool all_reached = true;
        for (const auto& controller : controllers) {
            if (controller.is_enabled) {
                double error = std::abs(controller.target_position - controller.current_position);
                if (error > 0.05) {
                    all_reached = false;
                    break;
                }
            }
        }
        
        // 如果所有电机都到达目标位置，提前结束
        if (all_reached && cycle > 30) {  // 至少运行30个循环
            std::cout << "\n✓ 所有电机已到达目标位置，提前结束控制循环" << std::endl;
            break;
        }
        
        // 等待10ms
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    // 最终状态显示
    display_motor_status(controllers, max_cycles, phase_name + " (最终状态)", target_velocity);
    
    return true;
}

// 禁用电机
void disable_motor(MotorController& controller) {
    if (controller.is_enabled && controller.motor_control && controller.motor) {
        try {
            std::cout << "禁用电机: " << controller.config.description << std::endl;
            controller.motor_control->disable(*controller.motor);
            controller.is_enabled = false;
        } catch (const std::exception& e) {
            std::cerr << "禁用电机失败 " << controller.config.description << ": " << e.what() << std::endl;
        }
    }
}

int main() {
    std::cout << "=== 升级版批量位置速度控制模式测试 ===" << std::endl;
    std::cout << "一次性控制所有电机运动到位，然后同时回零位" << std::endl;
    std::cout << "支持多种速度测试: ";
    for (size_t i = 0; i < test_velocities.size(); i++) {
        std::cout << test_velocities[i] << " rad/s";
        if (i < test_velocities.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;
    std::cout << "总共控制 " << motor_configs.size() << " 个电机" << std::endl;
    
    // 创建电机控制器数组
    std::vector<MotorController> controllers;
    controllers.reserve(motor_configs.size());
    
    // 初始化所有电机控制器
    std::cout << "\n=== 初始化所有电机 ===" << std::endl;
    int init_success_count = 0;
    for (const auto& config : motor_configs) {
        MotorController controller;
        controller.config = config;
        controller.is_enabled = false;
        
        if (init_motor_controller(controller)) {
            init_success_count++;
        }
        controllers.push_back(std::move(controller));
    }
    
    std::cout << "电机初始化完成: " << init_success_count << "/" << motor_configs.size() << " 成功" << std::endl;
    
    if (init_success_count == 0) {
        std::cerr << "没有电机初始化成功，退出测试" << std::endl;
        return -1;
    }
    
    // 等待所有电机稳定
    std::cout << "\n等待所有电机稳定..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // 逐个测试不同速度
    int total_tests = test_velocities.size();
    int successful_tests = 0;
    
    for (size_t speed_idx = 0; speed_idx < test_velocities.size(); speed_idx++) {
        double current_velocity = test_velocities[speed_idx];
        
        std::cout << "\n" << std::string(100, '=') << std::endl;
        std::cout << "=== 速度测试 " << (speed_idx + 1) << "/" << total_tests << ": " << current_velocity << " rad/s ===" << std::endl;
        
        // 批量运动到目标位置
        std::string move_phase = "速度 " + std::to_string(current_velocity) + " rad/s - 批量运动到目标位置";
        bool move_success = batch_move_to_target(controllers, 0.5, current_velocity, move_phase);
        
        // 等待2秒
        std::cout << "\n等待2秒..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));
        
        // 批量回零位
        std::string return_phase = "速度 " + std::to_string(current_velocity) + " rad/s - 批量回零位";
        bool return_success = batch_move_to_target(controllers, 0.0, current_velocity, return_phase);
        
        // 等待2秒
        std::cout << "\n等待2秒..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));
        
        if (move_success && return_success) {
            successful_tests++;
            std::cout << "✓ 速度 " << current_velocity << " rad/s 测试成功!" << std::endl;
        } else {
            std::cout << "✗ 速度 " << current_velocity << " rad/s 测试失败!" << std::endl;
        }
        
        // 如果不是最后一个速度测试，等待3秒
        if (speed_idx < test_velocities.size() - 1) {
            std::cout << "等待3秒后开始下一个速度测试..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(3));
        }
    }
    
    // 批量禁用所有电机
    std::cout << "\n" << std::string(100, '=') << std::endl;
    std::cout << "=== 批量禁用所有电机 ===" << std::endl;
    for (auto& controller : controllers) {
        disable_motor(controller);
    }
    
    // 测试结果总结
    std::cout << "\n" << std::string(100, '=') << std::endl;
    std::cout << "=== 测试结果总结 ===" << std::endl;
    std::cout << "总电机数: " << motor_configs.size() << std::endl;
    std::cout << "初始化成功: " << init_success_count << std::endl;
    std::cout << "测试速度数: " << total_tests << std::endl;
    std::cout << "成功测试: " << successful_tests << std::endl;
    std::cout << "失败测试: " << (total_tests - successful_tests) << std::endl;
    std::cout << "成功率: " << std::fixed << std::setprecision(1) 
              << (double)successful_tests / total_tests * 100 << "%" << std::endl;
    
    if (successful_tests == total_tests) {
        std::cout << "🎉 所有速度测试成功!" << std::endl;
    } else {
        std::cout << "⚠️  部分速度测试失败，请检查连接和配置" << std::endl;
    }
    
    return 0;
}
