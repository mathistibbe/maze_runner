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

        
        # Robot pose subscription
        self.create_subscription(
            Pose, "robot_pose", self.pose_callback, 10
        )
        self._current_pose = None
        self._real_world_yaw = None


        # Gyro rotation subscription
        self.rotation_sub = self.create_subscription(
            Float32, "rotation", self.rotation_callback, qos_profile_sensor_data
        )
        self.gyro_rotation = None

        # Publisher for velocity commands
        self._cmd_vel_pub = self.create_publisher(Twist, "cmd", 10)
        #self.rw_yaw_pub = self.create_publisher(Float32, "rw_yaw", 10)
        #self.timer = self.create_timer(1, self.timer_cb)

    
    def timer_cb(self):
        if self._current_pose is not None:
            yaw_msg = Float32()
            
            yaw_msg.data = self._get_yaw_from_quaternion(self._current_pose.orientation)
            self.rw_yaw_pub.publish(yaw_msg)

    def pose_callback(self, msg: Pose):
        """Update the current robot pose from subscriber."""
        new_yaw = self._get_yaw_from_quaternion(msg.orientation)
        #self.get_logger().info(f"Current pose: {new_yaw}")
        if self._real_world_yaw is None:
            if -0.3 < new_yaw < 0.3:
                self._current_pose = msg
                self._real_world_yaw = new_yaw
        else:
            if abs(self._angle_diff(self._real_world_yaw, new_yaw))< 0.3:
                self._current_pose = msg
                self._real_world_yaw = new_yaw

    def rotation_callback(self, msg):
        """Update the current rotation from subscriber."""
        self.gyro_rotation = msg.data

    def goal_callback(self, goal_request):
        """Accept or reject incoming goal request."""
        self.get_logger().info(f"Received goal: x={goal_request.x}, y={goal_request.y}")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Handle goal cancellation."""
        self.get_logger().info("Received cancel request")
        return CancelResponse.ACCEPT

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

    def pos_calulation(self, target_x, target_y):
        """Calculate the distance and angle to the target position."""
        if self._current_pose is None:
            return None, None

        dx = target_x - self._current_pose.position.x
        dy = target_y - self._current_pose.position.y
        distance = math.hypot(dx, dy)
        desired_yaw = math.atan2(-1*dx, dy)
        self.get_logger().info(f"dx: {dx}, dy: {dy}, distance: {distance}, desired_yaw: {desired_yaw}")

        #if distance > 0.5:
            # self.get_logger().info(f"Distance threshold not met: {distance:.2f} > 0.3")
        #    time.sleep(0.1)  # wait for the robot to stabilize
        #    return self.pos_calulation(target_x, target_y)

        return distance, desired_yaw


    def execute_callback(self, goal_handle):
        """Main loop that drives robot to target with separate rotation and translation phases."""
        self.get_logger().info("Executing goal...")

        target_x = goal_handle.request.y
        target_y = goal_handle.request.x

        self.get_logger().warn("Waiting for gyro rotation and robot pose...")
                        
        while self.gyro_rotation is None or self._current_pose is None:
            time.sleep(0.1)

        
       
        offset = self._angle_diff(target=self._real_world_yaw, current=self.gyro_rotation)

        feedback_msg = GoToPoint.Feedback()
        rate = 0.05

        distance, desired_yaw = self.pos_calulation(target_x, target_y)
        goal_gyro_yaw = desired_yaw - offset
            
        self.get_logger().info(f"Distance to goal: {distance:.2f}, Rotation: {self._angle_diff(goal_gyro_yaw, self.gyro_rotation):.2f}, Current gyro + Offset{self.normalize_angle(self.gyro_rotation + offset)}")

        # --- Phase 1: Rotate to face the target ---
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self.get_logger().info("Goal canceled")
                goal_handle.canceled()
                return GoToPoint.Result(success=False, message="Goal canceled")


            #current_yaw = self.normalize_angle(self.gyro_rotation + offset)
            
            

            #feedback_msg.distance_remaining = distance
            #goal_handle.publish_feedback(feedback_msg)

            # If facing the target (within threshold), break to next phase

            remaining_angle = self._angle_diff(goal_gyro_yaw, self.gyro_rotation)

            if abs(remaining_angle) < math.radians(10):
                twist = Twist()
                self._cmd_vel_pub.publish(twist)
                time.sleep(0.5)
                break

            # Rotate in place
            rotation_direction = remaining_angle / abs(remaining_angle)
            twist = Twist()
            twist.linear.x = 0.0 
            twist.angular.z = rotation_direction * math.radians(20) 
            self._cmd_vel_pub.publish(twist)
            time.sleep(rate)

        # --- Phase 2: Move forward towards the target ---
        moved_distance = 0.0
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self.get_logger().info("Goal canceled")
                goal_handle.canceled()
                return GoToPoint.Result(success=False, message="Goal canceled")

            

            #current_yaw = self.normalize_angle(self.gyro_rotation + offset)
            #desired_yaw = self.normalize_angle(math.atan2(dy, dx))
            #yaw_error = self.normalize_angle(self._angle_diff(desired_yaw, current_yaw))

            feedback_msg.distance_remaining = distance - moved_distance
            goal_handle.publish_feedback(feedback_msg)

            if distance - moved_distance < 0.05:
                # Stop robot
                twist = Twist()
                self._cmd_vel_pub.publish(twist)
                time.sleep(0.5)
                break

            # Move forward, but correct heading slightly if needed
            twist = Twist()
            twist.linear.x = 0.2
            moved_distance += twist.linear.x * rate
            #twist.angular.z = max(-0.3, min(0.3, 1.0 * yaw_error))  # Small correction
            self._cmd_vel_pub.publish(twist)
            time.sleep(rate)

        # Stop robot at the end
        twist = Twist()
        self._cmd_vel_pub.publish(twist)

        goal_handle.succeed()
        self.get_logger().info("Goal reached successfully")

        return GoToPoint.Result(success=True, message="Goal reached")


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
