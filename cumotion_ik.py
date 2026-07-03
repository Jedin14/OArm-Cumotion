#!/usr/bin/env python3

import sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest

class CuMotionIKClient(Node):
    def __init__(self):
        super().__init__('cumotion_ik_client')
        
        # MoveIt provides the /compute_ik service when running
        self.cli = self.create_client(GetPositionIK, '/compute_ik')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /compute_ik service... (Make sure you launched the cuMotion MoveIt node!)')
            
        self.req = GetPositionIK.Request()

    def send_ik_request(self, target_pose: PoseStamped, group_name="left_arm"):
        self.req.ik_request.group_name = group_name
        self.req.ik_request.pose_stamped = target_pose
        self.req.ik_request.timeout.sec = 1
        self.req.ik_request.avoid_collisions = True

        self.get_logger().info(f'Sending IK request for group {group_name}...')
        future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)
        
        response = future.result()
        if response.error_code.val == 1: # 1 is SUCCESS in MoveIt
            self.get_logger().info('IK Solution found!')
            # Extract joint positions
            joint_names = response.solution.joint_state.name
            joint_positions = response.solution.joint_state.position
            for name, pos in zip(joint_names, joint_positions):
                self.get_logger().info(f'  {name}: {pos:.3f} rad')
            return response.solution.joint_state
        else:
            self.get_logger().error(f'IK failed with error code: {response.error_code.val}')
            return None

def main(args=None):
    rclpy.init(args=args)
    ik_client = CuMotionIKClient()

    # Define a target pose for the end-effector
    target_pose = PoseStamped()
    target_pose.header.frame_id = "base_link"
    target_pose.header.stamp = ik_client.get_clock().now().to_msg()
    
    # Adjust this pose to a reachable point for your left_arm!
    target_pose.pose.position.x = 0.4
    target_pose.pose.position.y = 0.2
    target_pose.pose.position.z = 0.5
    target_pose.pose.orientation.w = 1.0

    # Send the request to cuMotion / MoveIt
    ik_client.send_ik_request(target_pose, group_name="left_arm")

    ik_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
