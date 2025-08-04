# go_to_point_action_server.py

import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from maze_interfaces.action import GoToPoint  # replace with your actual import
from geometry_msgs.msg import Pose  # or Odometry depending on your robot
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from geometry_msgs.msg import Quaternion
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
import numpy as np

import time


class GoToPointActionServer(Node):
    def __init__(self):
        super().__init__("go_to_point_action_server")

        # Create Action Server
        self._action_server = ActionServer(
            self,
            GoToPoint,
            "go_to_point",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )
        self.gyro_north_offset = None
        
        # Gyro rotation subscription
        self.rotation_sub = self.create_subscription(
            Float32, "rotation", self.rotation_callback, qos_profile_sensor_data
        )
        self.gyro_rotation = None

        # Publisher for velocity commands
        self._cmd_vel_pub = self.create_publisher(Twist, "cmd", 10)

    

    
    


    def execute_callback(self, goal_handle):
        """Main loop that drives robot to target with separate rotation and translation phases."""
        
        
        self.get_logger().info("Executing goal...")

        distance = int(goal_handle.request.x)
        heading = int(goal_handle.request.y)

        self.get_logger().warn("Waiting for gyro rotation and robot pose...")
                        
        while self.gyro_rotation is None:
            time.sleep(0.1)
        

        
       

        feedback_msg = GoToPoint.Feedback()
        rate = 0.05

        target_yaw = self.heading_to_angle(heading)

            

        # --- Phase 1: Rotate to face the target ---
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self.get_logger().info("Goal canceled")
                goal_handle.canceled()
                return GoToPoint.Result(success=False, message="Goal canceled")


            current_yaw = self.get_local_yaw()
            remaining_angle = self._angle_diff(target_yaw, current_yaw)
            
            if abs(remaining_angle) < math.radians(2):
                twist = Twist()
                self._cmd_vel_pub.publish(twist)
                time.sleep(0.2)
                break

            # Rotate in place
            rotation_direction = remaining_angle / abs(remaining_angle)
            twist = Twist()
            twist.linear.x = 0.0 
            twist.angular.z = 0.2 if remaining_angle > 0 else -0.2 
            self._cmd_vel_pub.publish(twist)
            time.sleep(rate)

        # --- Phase 2: Move forward towards the target ---
        moved_distance = 0.0
        total_distance = distance * 10/100
        while rclpy.ok() and moved_distance < total_distance:
            if goal_handle.is_cancel_requested:
                self.get_logger().info("Goal canceled")
                goal_handle.canceled()
                return GoToPoint.Result(success=False, message="Goal canceled")


            twist = Twist()
            twist.linear.x = 0.2
            moved_distance += 0.2 * rate
            self._cmd_vel_pub.publish(twist)
            time.sleep(rate)
            if total_distance - moved_distance < 0.02:
                break

        # Stop robot at the end
        twist = Twist()
        self._cmd_vel_pub.publish(twist)

        goal_handle.succeed()
        self.get_logger().info("Goal reached successfully")

        return GoToPoint.Result(success=True, message="Goal reached")

        

    def pos_calulation(self, target_x, target_y):
        """Calculate the distance and angle to the target position."""
        if self._current_pose is None:
            return None, None

        dx = target_x - self._current_pose.position.x
        dy = target_y - self._current_pose.position.y
        distance = math.hypot(dx, dy)
        desired_yaw = math.atan2(dy, dx)
        self.get_logger().info(f"dx: {dx}, dy: {dy}, distance: {distance}, desired_yaw: {desired_yaw}")

        #if distance > 0.5:
            # self.get_logger().info(f"Distance threshold not met: {distance:.2f} > 0.3")
        #    time.sleep(0.1)  # wait for the robot to stabilize
        #    return self.pos_calulation(target_x, target_y)

        return distance, desired_yaw

    
    def get_local_yaw(self):
        """Yaw relative to north at startup."""
        if self.gyro_rotation is None or self.gyro_north_offset is None:
            return None
        return self.normalize_angle(self.gyro_rotation - self.gyro_north_offset)

    def _get_yaw_from_quaternion(self, q: Quaternion):
        x, y, z, w = q.x, q.y, q.z, q.w
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _angle_diff(self, target, current):
        diff = target - current
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff

    def normalize_angle(self, angle):
        return np.arctan2(np.sin(angle),np.cos(angle))

    def goal_callback(self, goal_request):
        """Accept or reject incoming goal request."""
        self.get_logger().info(f"Received goal: x={goal_request.x}, y={goal_request.y}")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Handle goal cancellation."""

        self.get_logger().info("Received cancel request")
        return CancelResponse.ACCEPT

    def rotation_callback(self, msg):
        """Update the current rotation from subscriber."""
        self.gyro_rotation = msg.data
        # Set north offset on first reading
        if self.gyro_north_offset is None and self.gyro_rotation is not None:
            self.gyro_north_offset = self.gyro_rotation
            self.get_logger().info(f"Gyro north offset set: {self.gyro_north_offset:.2f}")


def main(args=None):
    rclpy.init(args=args)
    node = GoToPointActionServer()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()  # This will spin with multiple threads
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
