// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_cumotion/impl/cumotion_logger_bridge.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace cumotion
{

// Thread-local default ROS2 logger instance.
thread_local std::shared_ptr<CumotionLoggerBridge> default_logger =
  std::make_shared<CumotionLoggerBridge>();

// Thread-local current ROS2 logger instance.
thread_local std::shared_ptr<CumotionLoggerBridge> current_logger = default_logger;

// Initializes the bridge with a named ROS2 logger for message routing.
CumotionLoggerBridge::CumotionLoggerBridge(const std::string & logger_name)
: logger_(rclcpp::get_logger(logger_name)) {}

std::ostream & CumotionLoggerBridge::log(cumotion_lib::LogLevel log_level)
{
  if (log_level == cumotion_lib::LogLevel::FATAL) {
    fatal_error_message_ << prefix_;
    return fatal_error_message_;
  }

  // Clears the current stream buffer and prepares it for the next message.
  current_stream_.str("");
  current_stream_.clear();
  current_stream_ << prefix_;
  return current_stream_;
}

void CumotionLoggerBridge::finalizeLogMessage(cumotion_lib::LogLevel log_level)
{
  // Skip logging for FATAL errors here since handleFatalError() will log them with context.
  if (log_level == cumotion_lib::LogLevel::FATAL) {
    return;
  }

  std::string message = current_stream_.str();

  switch (log_level) {
    case cumotion_lib::LogLevel::ERROR:
      RCLCPP_ERROR_STREAM(logger_, message);
      break;
    case cumotion_lib::LogLevel::WARNING:
      RCLCPP_WARN_STREAM(logger_, message);
      break;
    case cumotion_lib::LogLevel::INFO:
      RCLCPP_INFO_STREAM(logger_, message);
      break;
    case cumotion_lib::LogLevel::VERBOSE:
      RCLCPP_DEBUG_STREAM(logger_, message);
      break;
    default:
      RCLCPP_INFO_STREAM(logger_, message);
      break;
  }
}

void CumotionLoggerBridge::handleFatalError()
{
  RCLCPP_FATAL_STREAM(
    logger_,
    "cuMotion fatal error occurred. " +
    fatal_error_message_.str());

  fatal_error_message_.str("");
  fatal_error_message_.clear();
}

void CumotionLoggerBridge::SetPrefix(const std::string & prefix)
{
  prefix_ = prefix;
}

void SetLogger(const std::shared_ptr<CumotionLoggerBridge> & logger)
{
  current_logger = logger ? logger : default_logger;
  cumotion_lib::SetLogger(current_logger);
}

void SetLogLevel(cumotion_lib::LogLevel log_level)
{
  cumotion_lib::SetLogLevel(log_level);
}

void SetLoggerPrefix(const std::string & prefix)
{
  current_logger->SetPrefix(prefix);
}

}  // namespace cumotion
}  // namespace isaac_ros
}  // namespace nvidia
