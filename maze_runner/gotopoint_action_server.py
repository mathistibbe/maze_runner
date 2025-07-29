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

        # Subscribe to position topic (Odometry or Pose)
        self._current_pose = None
        qos = QoSProfile(depth=10)
        self.create_subscription(
            Pose, "robot_pose", self.pose_callback, qos  # Replace with your topic
        )

        self.rotation_sub = self.create_subscription(  ##Gyro
            Float32, "rotation", self.rotation_callback, 10
        )
        self.gyro_rotation = None

        self._cmd_vel_pub = self.create_publisher(Twist, "cmd", 10)

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

    def pose_callback(self, msg: Pose):
        """Update the current robot pose from subscriber."""
        self._current_pose = msg

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

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


    def execute_callback(self, goal_handle):
        """Main loop that drives robot to target."""
        self.get_logger().info("Executing goal...")

        target_x = goal_handle.request.x
        target_y = goal_handle.request.y

        
        self.get_logger().warn("Waiting for gyro rotation and robot pose...")

        while self.gyro_rotation is None or self._current_pose is None:
            time.sleep(0.1)

        real_world_yaw = self._get_yaw_from_quaternion(self._current_pose.orientation)

        
        self.get_logger().info(f"Gyro rotation: {self.gyro_rotation}")

        offset = self.normalize_angle(real_world_yaw - self.gyro_rotation)

        feedback_msg = GoToPoint.Feedback()
        rate = 0.05

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self.get_logger().info("Goal canceled")
                goal_handle.canceled()
                return GoToPoint.Result(success=False, message="Goal canceled")

            if self._current_pose is None:
                self.get_logger().warn("Waiting for pose...")
                time.sleep(rate)
                continue

            # Compute distance to goal
            dx = target_x - self._current_pose.position.x
            dy = target_y - self._current_pose.position.y
            distance = math.hypot(dx, dy)

            # Calculate current yaw
            #q = self._current_pose.orientation
            # current_yaw = self._get_yaw_from_quaternion(q)
            current_yaw = self.normalize_angle(self.gyro_rotation + offset)

            # Desired heading
            desired_yaw = math.atan2(dy, dx)
            yaw_error = self._angle_diff(desired_yaw, current_yaw)

            # Publish feedback
            feedback_msg.distance_remaining = distance
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(
                f"Current Yaw: {current_yaw:.2f}, Desired Yaw: {desired_yaw:.2f}, Yaw Error: {yaw_error:.2f}"
            )

            if distance < 0.1:
                # Stop robot
                twist = Twist()
                self._cmd_vel_pub.publish(twist)
                time.sleep(rate)  # allow time to stop
                break

            # Proportional control
            linear_speed = min(0.2, 0.5 * distance)
            angular_speed = 1.0 * yaw_error  # Tune gain

            # Publish Twist command
            twist = Twist()
            twist.linear.x = linear_speed
            twist.angular.z = angular_speed
            self._cmd_vel_pub.publish(twist)

            # e.g., send velocity commands to move toward the target while adjusting orientation

            time.sleep(rate)

        twist = Twist()  # zero velocities
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
