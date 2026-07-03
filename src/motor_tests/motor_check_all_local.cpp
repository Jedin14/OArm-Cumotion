#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "include/openarm_hardware/canbus.hpp"
#include "include/openarm_hardware/motor.hpp"
#include "include/openarm_hardware/motor_control.hpp"

struct MotorStatusRow {
  std::string can_iface;
  int send_id;
  int recv_id;
  bool online;
  std::string mode;
  std::string note;
};

static std::string mode_name(int code) {
  switch (code) {
    case 1: return "MIT (阻抗控制)";
    case 2: return "POS_VEL (位置速度控制)";
    case 3: return "VEL (速度控制)";
    case 4: return "POS_FORCE (力位置控制)";
    default: return "未知/无";
  }
}

static void init_arm_on_bus(const std::string &iface,
                            std::vector<std::unique_ptr<Motor>> &motors,
                            std::unique_ptr<CANBus> &canbus,
                            std::unique_ptr<MotorControl> &mc) {
  canbus = std::make_unique<CANBus>(iface, CAN_MODE_CLASSIC);
  mc = std::make_unique<MotorControl>(*canbus);
  motors.clear();
  for (int i = 1; i <= 8; ++i) {
    motors.emplace_back(std::make_unique<Motor>(DM_Motor_Type::DM4310, i,
                                                i + 16));
    mc->addMotor(*motors.back());
  }
}

static bool ping_online(MotorControl &mc, Motor &m) {
  try {
    mc.controlMIT2(m, 0.0, 0.0, 0.0, 0.0, 0.0);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    mc.recv();
    return true;
  } catch (...) {
    return false;
  }
}

static void query_ctrl_mode(MotorControl &mc, Motor &m) {
  mc.queryMotorParam(m, DM_variable::CTRL_MODE);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  mc.recv_set_param_data();
}

int main() {
  std::vector<MotorStatusRow> rows;

  for (const std::string iface : {"can0", "can1"}) {
    try {
      std::unique_ptr<CANBus> canbus;
      std::unique_ptr<MotorControl> mc;
      std::vector<std::unique_ptr<Motor>> motors;
      init_arm_on_bus(iface, motors, canbus, mc);

      for (size_t idx = 0; idx < motors.size(); ++idx) {
        Motor &m = *motors[idx];
        MotorStatusRow row;
        row.can_iface = iface;
        row.send_id = static_cast<int>(m.SlaveID);
        row.recv_id = static_cast<int>(m.SlaveID + 16);
        row.online = false;
        row.mode = "N/A";
        row.note = "";

        try {
          // Step 1: detect online
          row.online = ping_online(*mc, m);

          // Step 2: if online, try read CTRL_MODE (with small retries)
          if (row.online) {
            int ctrl = -1;
            for (int t = 0; t < 3; ++t) {
              query_ctrl_mode(*mc, m);
              ctrl = m.getParam(static_cast<int>(DM_variable::CTRL_MODE));
              if (ctrl >= 0) break;
              std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            if (ctrl >= 0) {
              row.mode = mode_name(ctrl);
            } else {
              row.mode = "N/A";
              row.note = "无法读取控制模式";
            }
          } else {
            row.mode = "N/A";
            row.note = "离线或无响应";
          }
        } catch (...) {
          row.online = false;
          row.mode = "N/A";
          row.note = "离线或无响应";
        }

        rows.push_back(row);
      }

    } catch (const std::exception &e) {
      for (int i = 1; i <= 8; ++i) {
        rows.push_back(MotorStatusRow{iface, i, i + 16, false, "N/A",
                                      std::string("接口不可用: ") + e.what()});
      }
    }
  }

  // Print table
  std::cout << "================================================================================\n";
  std::cout << "                          电机控制模式检测结果 (本地)\n";
  std::cout << "================================================================================\n";
  std::cout << std::left << std::setw(8) << "CAN接口" << std::setw(10) << "发送ID"
            << std::setw(10) << "接收ID" << std::setw(8) << "状态" << std::setw(22)
            << "控制模式" << "备注" << std::endl;
  std::cout << "--------------------------------------------------------------------------------\n";

  int online_cnt = 0;
  int mit_cnt = 0, posvel_cnt = 0, vel_cnt = 0, posforce_cnt = 0;
  for (const auto &r : rows) {
    std::cout << std::left << std::setw(8) << r.can_iface << std::setw(10)
              << r.send_id << std::setw(10) << r.recv_id << std::setw(8)
              << (r.online ? "在线" : "离线") << std::setw(22) << r.mode
              << (r.note.empty() ? "" : r.note) << std::endl;
    if (r.online) online_cnt++;
    if (r.mode.rfind("MIT", 0) == 0) mit_cnt++;
    if (r.mode.rfind("POS_VEL", 0) == 0) posvel_cnt++;
    if (r.mode.rfind("VEL", 0) == 0 && r.mode != "POS_VEL (位置速度控制)") vel_cnt++;
    if (r.mode.rfind("POS_FORCE", 0) == 0) posforce_cnt++;
  }

  std::cout << "================================================================================\n\n";
  std::cout << "统计信息:\n";
  std::cout << "在线电机数量: " << online_cnt << "/" << rows.size() << "\n";
  std::cout << "MIT模式: " << mit_cnt << " 个\n";
  std::cout << "POS_VEL模式: " << posvel_cnt << " 个\n";
  std::cout << "VEL模式: " << vel_cnt << " 个\n";
  std::cout << "POS_FORCE模式: " << posforce_cnt << " 个\n";
  std::cout << "================================================================================\n";

  return 0;
} 