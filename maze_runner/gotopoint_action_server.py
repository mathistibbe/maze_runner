# go_to_point_action_server.py

import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from maze_interfaces.action import GoToPoint  # replace with your actual import
from geometry_msgs.msg import Pose, Point  # or Odometry depending on your robot
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from geometry_msgs.msg import Quaternion
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from .map_creation import RESOLUTION_FACTOR
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
            execute_callback=self.old_execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )
        self.gyro_north_offset = None
        self.robot_pos = None

        self.declare_parameter("initial_heading", 0)
        self.current_heading = self.get_parameter("initial_heading").value

        self._accept_cancel = True  # Flag to control cancel acceptance

        self.robot_pose_sub = self.create_subscription(
            Pose,
            "robot_pose",  # replace with your actual topic name
            self.robot_pose_callback,
            QoSProfile(depth=10)
        )
        #self.clean_robot_pose_sub = self.create_subscription(
        #    Point,
        #    "clean_robot_pose",  # replace with your actual topic name
        #    self.robot_pose_callback,
        #    QoSProfile(depth=10)
        #)
    


        # Gyro rotation subscription
        self.rotation_sub = self.create_subscription(
            Float32, "rotation", self.rotation_callback, qos_profile_sensor_data
        )
        self.gyro_rotation = None

        # Publisher for velocity commands
        self._cmd_vel_pub = self.create_publisher(Twist, "cmd", 10)

    def goal_callback(self, goal_request):
        """Accept or reject incoming goal request."""
        self.get_logger().info(f"Received goal: x={goal_request.x}, y={goal_request.y}")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Handle goal cancellation."""
          # Allow time for reverse
        #return CancelResponse.ACCEPT
        if self._accept_cancel:
            self.get_logger().info("Cancel request accepted")
            return CancelResponse.ACCEPT
        else:
            self.get_logger().info("Cancel request rejected")
            # Optionally, you can log or handle the rejection case
            return CancelResponse.REJECT
    

    def rotation_callback(self, msg):
        """Update the current rotation from subscriber."""
        self.gyro_rotation = msg.data
        if self.gyro_north_offset is None and self.gyro_rotation is not None:
            self.get_logger().info(f"Current heading: {self.current_heading}")
            self.gyro_north_offset = self.gyro_rotation
            self.gyro_north_offset += self.heading_to_angle(self.current_heading)
            self.get_logger().info(f"Gyro north offset set (adjusted): {self.gyro_north_offset:.2f}")

    def robot_pose_callback(self, msg):
        """Update the current robot pose from subscriber."""
        self.robot_pos = (msg.position.x, msg.position.y)

    def heading_to_angle(self, heading):
        """Convert cardinal heading to yaw angle in radians (relative to north)."""
        mapping = {
            0: 0.0, # North
            1: math.pi / 4, # North-East
            2: math.pi/2, # East
            3: 3 * math.pi / 4, # South-East
            4: math.pi, # South
            5: -3 * math.pi / 4, # South-West
            6: -math.pi / 2, # West
            7: -math.pi / 4 # North-West
        }
        #mapping = [0.0, -math.pi / 2, math.pi, math.pi / 2]
        return mapping[heading]

    
    
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
    
    def execute_callback(self, goal_handle):
        self.get_logger().info("Executing goal...")

        distance = int(goal_handle.request.x)
        heading = int(goal_handle.request.y)

        # Wait for initial data
        self.get_logger().warn("Waiting for gyro rotation and robot pose...")
        while self.gyro_rotation is None or self.robot_pos is None:
            time.sleep(0.1)

        target_yaw = self.heading_to_angle(heading)
        start_x, start_y = self.robot_pos
        target_distance = distance * RESOLUTION_FACTOR / 100  # cm to meters
        moved_distance_estimate = 0.0
        rate = 0.05  # 50Hz

        self._accept_cancel = True
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                stop_twist = Twist()
                stop_twist.linear.x = 0.0
                stop_twist.angular.z = 0.0
                self._cmd_vel_pub.publish(stop_twist)
                time.sleep(0.05)  # Allow time for stop command to take effect
                reverse_twist = Twist()
                reverse_twist.linear.x = -0.3  # Reverse speed
                self._cmd_vel_pub.publish(reverse_twist)
                time.sleep(0.7) 
                self.get_logger().info("Goal canceled")
                goal_handle.canceled()
                return GoToPoint.Result(success=False, message="Goal canceled")

            current_yaw = self.get_local_yaw()
            if current_yaw is None or self.robot_pos is None:
                self.get_logger().warn("Waiting for pose or heading...")
                time.sleep(0.1)
                continue

            current_x, current_y = self.robot_pos
            dx = current_x - start_x
            dy = current_y - start_y
            moved_distance_pos = math.hypot(dx, dy)

            # Stop conditions
            if moved_distance_pos >= target_distance:
                break
            elif moved_distance_estimate > 3 * moved_distance_pos:
                if moved_distance_estimate > target_distance:
                    self.get_logger().info("Using estimated distance to stop.")
                    break

            # --- Smooth rotation while moving forward ---
            angle_error = self._angle_diff(target_yaw, current_yaw)

            # Angular velocity proportional to error
            angular_z = np.clip(-2.0 * angle_error, -1.5, 1.5)

            # Reduce linear speed if angular error is large
            linear_speed = max(0.3, min(0.5, (target_distance - moved_distance_estimate)* 0.8)) if abs(angle_error) < math.radians(15) else 0.3

            twist = Twist()
            twist.linear.x = linear_speed
            twist.angular.z = angular_z
            self._cmd_vel_pub.publish(twist)

            moved_distance_estimate += twist.linear.x * rate
            time.sleep(rate)

        # Stop robot
        self._cmd_vel_pub.publish(Twist())
        goal_handle.succeed()
        self.get_logger().info("Goal reached successfully")
        return GoToPoint.Result(success=True, message="Goal reached")

    def old_execute_callback(self, goal_handle):
        """Main loop that drives robot to target with separate rotation and translation phases."""
        
        
        self.get_logger().info("Executing goal...")

        distance = int(goal_handle.request.x)
        heading = int(goal_handle.request.y)

        self.get_logger().warn("Waiting for gyro rotation and robot pose...")
                        
        while self.gyro_rotation is None:
            time.sleep(0.1)
        

        
       

        feedback_msg = GoToPoint.Feedback()
        rate = 0.05  # 50 Hz

        target_yaw = self.heading_to_angle(heading)

            

        # --- Phase 1: Rotate to face the target ---
        self._accept_cancel = False  # Disable cancel during rotation phase
        while rclpy.ok():
            current_yaw = self.get_local_yaw()
            remaining_angle = self._angle_diff(target_yaw, current_yaw)
            
            if abs(remaining_angle) < math.radians(2):
                twist = Twist()
                self._cmd_vel_pub.publish(twist)
                time.sleep(0.1)
                break

            # Rotate in place
            # rotation_direction = remaining_angle / abs(remaining_angle)
            twist = Twist()
            twist.linear.x = 0.0 
            twist.angular.z = -0.7 if remaining_angle > 0 else 0.7
            self._cmd_vel_pub.publish(twist)
            time.sleep(rate)

        # --- Phase 2: Move forward towards the target ---
        moved_distance_estimate = 0.0
        self._accept_cancel = True  # Re-enable cancel after rotation
        while self.robot_pos is None:
            self.get_logger().warn("Waiting for initial robot position...")
            time.sleep(0.1)

        start_x, start_y = self.robot_pos
        target_distance = distance * RESOLUTION_FACTOR / 100 # cm
        while rclpy.ok():

            if goal_handle.is_cancel_requested:
                self.get_logger().info("Goal canceled in movement phase")
                self._cmd_vel_pub.publish(Twist())  # Stop robot
                reverse_twist = Twist()
                time.sleep(0.1)  # Allow time for stop command to take effect
                reverse_twist.linear.x = -0.2  # Reverse speed
                self._cmd_vel_pub.publish(reverse_twist)
                time.sleep(0.5)
                self._cmd_vel_pub.publish(Twist())  # Stop robot
                goal_handle.canceled()
                return GoToPoint.Result(success=False, message="Goal canceled")

            if self.robot_pos is None:
                self.get_logger().warn("Waiting for robot position...")
                time.sleep(0.1)
                continue
            
            current_x, current_y = self.robot_pos
            dx = current_x - start_x
            dy = current_y - start_y
            
            moved_distance_pos = math.hypot(dx, dy)
           

            
            if moved_distance_pos >= target_distance:
                break
            elif moved_distance_estimate > 3* moved_distance_pos:
                # Robot tracking is off use estimated distance
                if moved_distance_estimate > target_distance:
                    self.get_logger().info("Estimated distance exceeded target, stopping.")
                    break
            
              # Assuming 0.4 m/s speed and 0.05s rate
            twist = Twist()
            twist.linear.x = max(0.3, min(1.0, target_distance - moved_distance_estimate * 0.8) )# Forward speed
            moved_distance_estimate += twist.linear.x * rate
            self._cmd_vel_pub.publish(twist)
            self.get_logger().info(f"Postion change: {moved_distance_pos:.2f}m, Estimated distance: {moved_distance_estimate}m, Target: {target_distance:.2f}m")
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
