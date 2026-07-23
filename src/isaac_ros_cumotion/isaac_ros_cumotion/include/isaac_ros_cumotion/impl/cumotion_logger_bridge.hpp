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

#ifndef ISAAC_ROS_CUMOTION__IMPL__CUMOTION_LOGGER_BRIDGE_HPP_
#define ISAAC_ROS_CUMOTION__IMPL__CUMOTION_LOGGER_BRIDGE_HPP_

#include <cumotion/cumotion.h>

#include <memory>
#include <sstream>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "isaac_ros_cumotion/impl/utils.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace cumotion
{

/**
 * Custom logger that routes cuMotion log messages through ROS2's logging infrastructure.
 */
class CumotionLoggerBridge : public cumotion_lib::Logger
{
public:
  /**
   * Constructs a ROS2 logger with the specified name. The logger_name parameter determines the
   * name that will appear in ROS2 log messages, defaulting to "cumotion_logger_bridge" if not
   * specified.
   */
  explicit CumotionLoggerBridge(const std::string & logger_name = "cumotion_logger_bridge");

  /**
   * Sets a prefix to be prepended to all log messages. The prefix string will be added to the
   * beginning of each log message, which is useful for identifying the source of messages when
   * multiple loggers are active.
   */
  void SetPrefix(const std::string & prefix);

private:
  /**
   * Returns an output stream suitable for logging messages at the given log level. The log_level
   * parameter determines which internal buffer is used - fatal messages use a separate buffer to
   * ensure they are properly captured even if an exception occurs. The returned stream can be
   * used to write log content.
   */
  std::ostream & log(cumotion_lib::LogLevel log_level) override;

  /**
   * Routes the logged message to the appropriate ROS2 logging level. This method is called after
   * log content has been written to finalize the message. The log_level parameter is mapped from
   * cuMotion levels (FATAL, ERROR, WARNING, INFO, VERBOSE) to corresponding ROS2 severity levels.
   */
  void finalizeLogMessage(cumotion_lib::LogLevel log_level) override;

  /**
   * Handles fatal errors by logging them through ROS2.
   */
  void handleFatalError() override;
  // ROS2 logger instance for outputting messages.
  rclcpp::Logger logger_;

  // Optional prefix prepended to all log messages.
  std::string prefix_;

  // Buffer for accumulating fatal error messages.
  std::ostringstream fatal_error_message_;

  // Temporary buffer for non-fatal log messages.
  std::ostringstream current_stream_;
};

/**
 * Installs a ROS2 logger for the current thread. Pass nullptr to restore the default logger.
 */
void SetLogger(const std::shared_ptr<CumotionLoggerBridge> & logger);

/**
 * Sets the log level threshold for cuMotion. Only messages with severity equal to or higher than
 * the specified log_level will be output.
 */
void SetLogLevel(cumotion_lib::LogLevel log_level);

/**
 * Sets the prefix for the thread-local default ROS2 logger. The prefix string will be prepended
 * to all log messages from the current thread's logger.
 */
void SetLoggerPrefix(const std::string & prefix);

}  // namespace cumotion
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_CUMOTION__IMPL__CUMOTION_LOGGER_BRIDGE_HPP_
