// Copyright 2025 Enactic, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <linux/can.h>

#include <cstdint>
#include <cstring>  // for memcpy
#include <iostream>
#include <map>
#include <vector>

#include "dm_motor.hpp"
#include "dm_motor_constants.hpp"

namespace openarm::damiao_motor {
// Forward declarations
class Motor;

struct ParamResult {
    int rid;
    double value;
    bool valid;
};

struct StateResult {
    double position;
    double velocity;
    double torque;
    int t_mos;
    int t_rotor;
    bool valid;
    // The status nibble the motor sends in byte 0 of every feedback frame:
    // 0 disabled, 1 enabled, 8 over-voltage, 9 under-voltage, 0xA
    // over-current, 0xB MOS over-temperature, 0xC rotor over-temperature,
    // 0xD lost communication, 0xE overload. -1 when it could not be read.
    //
    // Byte 0 was being skipped entirely, so nothing above this library
    // could tell a motor that is driving from one that is merely
    // reporting. See parse_motor_state_data for how it is validated
    // rather than assumed.
    int status = -1;
};

struct CANPacket {
    uint32_t send_can_id;
    std::vector<uint8_t> data;
};

struct MITParam {
    double kp;
    double kd;
    double q;
    double dq;
    double tau;
};

class CanPacketEncoder {
public:
    static CANPacket create_enable_command(const Motor& motor);
    static CANPacket create_disable_command(const Motor& motor);
    static CANPacket create_set_zero_command(const Motor& motor);
    // 0xFB, the same shape as the others. A Damiao motor latches a fault
    // -- overload, over-current, either over-temperature -- and produces
    // no torque until it is cleared, while its encoder keeps reporting
    // normally. Without this there was no way to clear one short of
    // power-cycling the arm.
    static CANPacket create_clear_error_command(const Motor& motor);
    static CANPacket create_mit_control_command(const Motor& motor, const MITParam& mit_param);
    static CANPacket create_query_param_command(const Motor& motor, int RID);
    static CANPacket create_refresh_command(const Motor& motor);

private:
    static std::vector<uint8_t> pack_mit_control_data(MotorType motor_type,
                                                      const MITParam& mit_param);
    static std::vector<uint8_t> pack_query_param_data(uint32_t send_can_id, int RID);
    static std::vector<uint8_t> pack_command_data(uint8_t cmd);

    static double limit_min_max(double x, double min, double max);
    static uint16_t double_to_uint(double x, double x_min, double x_max, int bits);
};

class CanPacketDecoder {
public:
    static StateResult parse_motor_state_data(const Motor& motor, const std::vector<uint8_t>& data);
    static ParamResult parse_motor_param_data(const std::vector<uint8_t>& data);

private:
    static double uint_to_double(uint16_t x, double min, double max, int bits);
    static float uint8s_to_float(const std::array<uint8_t, 4>& bytes);
    static uint32_t uint8s_to_uint32(uint8_t byte1, uint8_t byte2, uint8_t byte3, uint8_t byte4);
    static bool is_in_ranges(int number);
};

}  // namespace openarm::damiao_motor
