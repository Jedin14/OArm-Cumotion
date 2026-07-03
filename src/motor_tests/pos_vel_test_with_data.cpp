// Copyright 2025 OpenArm Control Test
//
// 位置速度控制模式测试程序 - 带数据读取功能
// 测试can0上的001号电机转动0.5rad，并实时显示位置/速度/扭矩数据

#include <iostream>
#include <chrono>
#include <thread>
#include <unistd.h>
#include <iomanip>

// 包含openarm_hardware的头文件
#include "openarm_hardware/canbus.hpp"
#include "openarm_hardware/motor.hpp"
#include "openarm_hardware/motor_control.hpp"

int main() {
    std::cout << "=== 位置速度控制模式测试 (带数据读取) ===" << std::endl;
    std::cout << "测试can0上的001号电机转动0.5rad，并实时显示电机数据" << std::endl;

    try {
        // 初始化CAN总线
        std::cout << "初始化CAN总线..." << std::endl;
        CANBus canbus("can0", CAN_MODE_CLASSIC);
        
        // 初始化电机控制
        MotorControl motor_control(canbus);
        
        // 创建电机对象 - 001号电机
        // 假设使用DM4310电机类型，SlaveID=1, MasterID=0x11
        Motor motor(DM_Motor_Type::DM4310, 1, 0x11);
        
        // 将电机添加到控制器
        motor_control.addMotor(motor);
        
        // 启用电机
        std::cout << "启用电机..." << std::endl;
        motor_control.enable(motor);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        // 切换到位置速度控制模式
        // std::cout << "切换到位置速度控制模式..." << std::endl;
        // bool mode_switched = motor_control.switchControlMode(motor, Control_Type::POS_VEL);
        // if (!mode_switched) {
        //     std::cerr << "警告: 无法切换到位置速度控制模式" << std::endl;
        // }
        // std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        // 设置目标位置和速度
        double target_position = 0.5;  // 目标位置: 0.5 rad
        double target_velocity = 0.2;  // 目标速度: 0.2 rad/s
        
        std::cout << "开始位置速度控制..." << std::endl;
        std::cout << "目标位置: " << target_position << " rad" << std::endl;
        std::cout << "目标速度: " << target_velocity << " rad/s" << std::endl;
        std::cout << std::endl;
        
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
        for (int i = 0; i < 50; i++) {  // 控制50次，每次间隔100ms
            // 发送控制命令
            motor_control.controlPosVel(motor, target_position, target_velocity);
            
            // 读取电机状态数据
            double actual_position = motor.getPosition();
            double actual_velocity = motor.getVelocity();
            double actual_torque = motor.getTorque();
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
            
            // 等待100ms
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        std::cout << std::endl;
        std::cout << "控制阶段完成，开始停止阶段..." << std::endl;
        
        // 停止电机 - 目标位置设为0
        for (int i = 0; i < 20; i++) {  // 停止20次，每次间隔100ms
            motor_control.controlPosVel(motor, 0.0, 0.0);
            
            // 读取电机状态数据
            double actual_position = motor.getPosition();
            double actual_velocity = motor.getVelocity();
            double actual_torque = motor.getTorque();
            
            // 显示停止阶段数据
            if (i % 5 == 0) {
                std::cout << "停止 " << std::setw(3) << (i+1) << "/20"
                          << " - 位置: " << std::fixed << std::setprecision(3) << actual_position
                          << " rad, 速度: " << std::fixed << std::setprecision(3) << actual_velocity
                          << " rad/s, 扭矩: " << std::fixed << std::setprecision(3) << actual_torque
                          << " Nm" << std::endl;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        // 禁用电机
        std::cout << "禁用电机..." << std::endl;
        motor_control.disable(motor);
        
        std::cout << "测试完成!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
