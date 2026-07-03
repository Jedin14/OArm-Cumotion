// Copyright 2025 OpenArm Control Test
//
// 增强版位置速度控制模式测试程序
// 测试can0-can1，1-8号电机的逐个运动，每个电机转动0-0.5rad

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
};

// 电机配置数组
const std::vector<MotorConfig> motor_configs = {
    {"can0", 1, DM_Motor_Type::DM4310, 0x01, 0x11, "Right Arm Joint 1", false},
    {"can0", 2, DM_Motor_Type::DM4310, 0x02, 0x12, "Right Arm Joint 2", false},
    {"can0", 3, DM_Motor_Type::DM4310, 0x03, 0x13, "Right Arm Joint 3", false},
    {"can0", 4, DM_Motor_Type::DM4310, 0x04, 0x14, "Right Arm Joint 4", false},
    {"can0", 5, DM_Motor_Type::DM4310, 0x05, 0x15, "Right Arm Joint 5", false},
    {"can0", 6, DM_Motor_Type::DM4310, 0x06, 0x16, "Right Arm Joint 6", false},
    {"can0", 7, DM_Motor_Type::DM4310, 0x07, 0x17, "Right Arm Joint 7", false},
    {"can0", 8, DM_Motor_Type::DM4310, 0x08, 0x18, "Right Arm Gripper", true},  // 反向
    {"can1", 1, DM_Motor_Type::DM4310, 0x01, 0x11, "Left Arm Joint 1", true},   // 反向
    {"can1", 2, DM_Motor_Type::DM4310, 0x02, 0x12, "Left Arm Joint 2", true},   // 反向
    {"can1", 3, DM_Motor_Type::DM4310, 0x03, 0x13, "Left Arm Joint 3", true},   // 反向
    {"can1", 4, DM_Motor_Type::DM4310, 0x04, 0x14, "Left Arm Joint 4", false},
    {"can1", 5, DM_Motor_Type::DM4310, 0x05, 0x15, "Left Arm Joint 5", true},   // 反向
    {"can1", 6, DM_Motor_Type::DM4310, 0x06, 0x16, "Left Arm Joint 6", true},   // 反向
    {"can1", 7, DM_Motor_Type::DM4310, 0x07, 0x17, "Left Arm Joint 7", true},   // 反向
    {"can1", 8, DM_Motor_Type::DM4310, 0x08, 0x18, "Left Arm Gripper", true}    // 反向
};

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
        
        // 启用电机
        std::cout << "启用电机: " << controller.config.description << std::endl;
        controller.motor_control->enable(*controller.motor);
        controller.is_enabled = true;
        
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "初始化电机失败 " << controller.config.description << ": " << e.what() << std::endl;
        controller.is_enabled = false;
        return false;
    }
}

// 测试单个电机
bool test_single_motor(MotorController& controller) {
    std::cout << "\n=== 测试电机: " << controller.config.description << " ===" << std::endl;
    std::cout << "CAN接口: " << controller.config.can_interface << ", 电机ID: " << controller.config.motor_id;
    if (controller.config.reverse_direction) {
        std::cout << " (反向旋转)";
    }
    std::cout << std::endl;
    
    try {
        // 设置目标位置和速度
        double target_position = controller.config.reverse_direction ? -0.5 : 0.5;  // 根据配置设置方向
        double target_velocity = 1.0;  // 目标速度: 1.0 rad/s
        
        std::cout << "开始位置速度控制..." << std::endl;
        std::cout << "目标位置: " << target_position << " rad" << std::endl;
        std::cout << "目标速度: " << target_velocity << " rad/s" << std::endl;
        
        // 显示表头
        std::cout << std::setw(6) << "循环" 
                  << std::setw(12) << "目标位置" 
                  << std::setw(12) << "实际位置" 
                  << std::setw(12) << "目标速度" 
                  << std::setw(12) << "实际速度" 
                  << std::setw(12) << "实际扭矩" 
                  << std::setw(12) << "位置误差" << std::endl;
        std::cout << std::setw(6) << "---" 
                  << std::setw(12) << "--------" 
                  << std::setw(12) << "--------" 
                  << std::setw(12) << "--------" 
                  << std::setw(12) << "--------" 
                  << std::setw(12) << "--------" 
                  << std::setw(12) << "--------" << std::endl;
        
        // 执行位置速度控制
        for (int i = 0; i < 50; i++) {  // 控制50次，每次间隔10ms
            // 发送控制命令
            controller.motor_control->controlPosVel(*controller.motor, target_position, target_velocity);
            
            // 读取电机状态数据
            double actual_position = controller.motor->getPosition();
            double actual_velocity = controller.motor->getVelocity();
            double actual_torque = controller.motor->getTorque();
            double position_error = target_position - actual_position;
            
            // 显示数据（每5次显示一次，避免刷屏）
            if (i % 5 == 0) {
                std::cout << std::setw(6) << (i+1)
                          << std::setw(12) << std::fixed << std::setprecision(3) << target_position
                          << std::setw(12) << std::fixed << std::setprecision(3) << actual_position
                          << std::setw(12) << std::fixed << std::setprecision(3) << target_velocity
                          << std::setw(12) << std::fixed << std::setprecision(3) << actual_velocity
                          << std::setw(12) << std::fixed << std::setprecision(3) << actual_torque
                          << std::setw(12) << std::fixed << std::setprecision(3) << position_error
                          << std::endl;
            }
            
            // 等待10ms
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        // 停止电机 - 目标位置设为0
        for (int i = 0; i < 20; i++) {  // 停止20次，每次间隔10ms
            controller.motor_control->controlPosVel(*controller.motor, 0.0, 0.0);
            
            // 读取电机状态数据
            double actual_position = controller.motor->getPosition();
            double actual_velocity = controller.motor->getVelocity();
            double actual_torque = controller.motor->getTorque();
            
            // 显示停止阶段数据
            if (i % 5 == 0) {
                std::cout << "停止 " << std::setw(3) << (i+1) << "/20"
                          << " - 位置: " << std::fixed << std::setprecision(3) << actual_position
                          << " rad, 速度: " << std::fixed << std::setprecision(3) << actual_velocity
                          << " rad/s, 扭矩: " << std::fixed << std::setprecision(3) << actual_torque
                          << " Nm" << std::endl;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        std::cout << "✓ " << controller.config.description << " 测试完成!" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "✗ " << controller.config.description << " 测试失败: " << e.what() << std::endl;
        return false;
    }
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
    std::cout << "=== 增强版位置速度控制模式测试 ===" << std::endl;
    std::cout << "测试can0-can1，1-8号电机的逐个运动" << std::endl;
    std::cout << "每个电机转动0-0.5rad，并实时显示电机数据" << std::endl;
    std::cout << "总共测试 " << motor_configs.size() << " 个电机" << std::endl;
    
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
    
    // 逐个测试每个电机
    int success_count = 0;
    for (size_t i = 0; i < controllers.size(); i++) {
        auto& controller = controllers[i];
        
        if (!controller.is_enabled) {
            std::cout << "\n跳过未初始化的电机: " << controller.config.description << std::endl;
            continue;
        }
        
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "测试进度: " << (i+1) << "/" << controllers.size() << std::endl;
        
        if (test_single_motor(controller)) {
            success_count++;
        }
        
        // 电机间等待时间
        if (i < controllers.size() - 1) {
            std::cout << "等待2秒后测试下一个电机..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }
    
    // 批量禁用所有电机
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "=== 批量禁用所有电机 ===" << std::endl;
    for (auto& controller : controllers) {
        disable_motor(controller);
    }
    
    // 测试结果总结
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "=== 测试结果总结 ===" << std::endl;
    std::cout << "总电机数: " << motor_configs.size() << std::endl;
    std::cout << "初始化成功: " << init_success_count << std::endl;
    std::cout << "测试成功: " << success_count << std::endl;
    std::cout << "测试失败: " << (init_success_count - success_count) << std::endl;
    std::cout << "成功率: " << std::fixed << std::setprecision(1) 
              << (double)success_count / init_success_count * 100 << "%" << std::endl;
    
    if (success_count == init_success_count) {
        std::cout << "🎉 所有电机测试成功!" << std::endl;
    } else {
        std::cout << "⚠️  部分电机测试失败，请检查连接和配置" << std::endl;
    }
    
    return 0;
}
