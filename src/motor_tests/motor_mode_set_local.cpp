#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "include/openarm_hardware/canbus.hpp"
#include "include/openarm_hardware/motor.hpp"
#include "include/openarm_hardware/motor_control.hpp"

static int parse_mode(const std::string &m) {
  if (m == "MIT" || m == "1") return 1;
  if (m == "POS_VEL" || m == "2") return 2;
  if (m == "VEL" || m == "3") return 3;
  if (m == "POS_FORCE" || m == "4") return 4;
  return -1;
}

static std::string mode_name(int code) {
  switch (code) {
    case 1: return "MIT (阻抗控制)";
    case 2: return "POS_VEL (位置速度控制)";
    case 3: return "VEL (速度控制)";
    case 4: return "POS_FORCE (力位置控制)";
    default: return "未知/无";
  }
}

static std::vector<int> parse_ids(const std::string &s) {
  std::vector<int> ids;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    try {
      int id = std::stoi(item);
      if (id >= 1 && id <= 8) ids.push_back(id);
    } catch (...) {}
  }
  return ids;
}

static bool verify_mode(MotorControl &mc, Motor &m, int expected_mode) {
  for (int t = 0; t < 5; ++t) {
    mc.queryMotorParam(m, DM_variable::CTRL_MODE);
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    mc.recv_set_param_data();
    int val = m.getParam(static_cast<int>(DM_variable::CTRL_MODE));
    if (val == expected_mode) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  return false;
}

int main(int argc, char *argv[]) {
  std::string iface = "can0";
  std::vector<int> ids;
  bool all = false;
  bool save = false;
  int mode = -1;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "-i" || arg == "--interface") && i + 1 < argc) {
      iface = argv[++i];
    } else if ((arg == "-m" || arg == "--mode") && i + 1 < argc) {
      mode = parse_mode(argv[++i]);
      if (mode == -1) {
        std::cerr << "错误: 无效的控制模式\n";
        return 1;
      }
    } else if (arg == "-a" || arg == "--all") {
      all = true;
    } else if (arg == "--ids" && i + 1 < argc) {
      ids = parse_ids(argv[++i]);
    } else if (arg == "--save") {
      save = true;
    } else if (arg == "-h" || arg == "--help") {
      std::cout << "用法: " << argv[0] << " -m MIT|POS_VEL|VEL|POS_FORCE [-i canX] [--ids 1,2,3 | -a] [--save]\n";
      return 0;
    }
  }

  if (mode == -1) {
    std::cerr << "错误: 必须指定控制模式 -m\n";
    return 1;
  }
  if (!all && ids.empty()) {
    std::cerr << "错误: 需使用 --ids 或 -a\n";
    return 1;
  }
  if (all) {
    ids.clear();
    for (int i = 1; i <= 8; ++i) ids.push_back(i);
  }

  std::cout << "=== 设置电机控制模式 (本地) ===\n";
  std::cout << "CAN接口: " << iface << "\n";
  std::cout << "控制模式: " << mode_name(mode) << "\n";
  std::cout << "电机ID: ";
  for (size_t i = 0; i < ids.size(); ++i) {
    std::cout << ids[i] << (i + 1 < ids.size() ? ", " : "\n");
  }
  std::cout << "保存到电机: " << (save ? "是" : "否") << "\n\n";

  try {
    auto canbus = std::make_unique<CANBus>(iface, CAN_MODE_CLASSIC);
    auto mc = std::make_unique<MotorControl>(*canbus);

    int success = 0;
    for (int id : ids) {
      Motor m(DM_Motor_Type::DM4310, static_cast<uint16_t>(id), static_cast<uint16_t>(id + 16));
      mc->addMotor(m);

      std::cout << "  设置电机 " << id << " -> " << mode_name(mode) << "...";
      bool ok = mc->switchControlMode(m, static_cast<Control_Type>(mode));
      if (!ok) {
        std::cout << " 失败" << std::endl;
        continue;
      }

      if (save) {
        std::cout << " 保存...";
        mc->save_motor_param(m);
      }

      bool verified = verify_mode(*mc, m, mode);
      std::cout << (verified ? " 完成" : " 未验证") << std::endl;
      if (verified) success++;
    }

    std::cout << "\n=== 设置完成 ===\n";
    std::cout << "成功: " << success << "/" << ids.size() << std::endl;
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "错误: " << e.what() << std::endl;
    return 1;
  }
} 