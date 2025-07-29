import rclpy
from rclpy.node import Node
import time
from std_msgs.msg import String, Bool, Int32, Float32
from geometry_msgs.msg import Twist
import threading
import math
from maze_interfaces.msg import Path, Index


class MovementController(Node):
    def __init__(self):
        super().__init__("movement_controller")
        self.cmd_publisher = self.create_publisher(Twist, "cmd", 10)

        self.movement_subscriber = self.create_subscription(
            String, "execute_movement", self.movement_callback, 10
        )
        self.movement_subscriber = self.create_subscription(
            Int32, "movement_tiles", self.move_tiles_callback, 10
        )
        self.rotation_sub = self.create_subscription(
            Float32, "rotation", self.rotation_cb, 10
        )
        self.path_subscriber = self.create_subscription(
            Path, "follow_path", self.path_callback, 10
        )

        self.EXECUTION_FREQ = 150
        self.FORWARD_SPEED = 0.15  # m/s # 0.15 y 0.2 b
        self.ROTATION_SPEED = 0.25
        self.robot_orientation = None
        self.path = None
        self.step_count = None
        self.heading = None

    def rotation_cb(self, msg):
        self.robot_orientation = msg.data

    def movement_callback(self, msg: String):
        command = msg.data.lower()
        self.get_logger().info(f"Received movement command: {command}")

        if command == "forward":
            self.send_forward()
        elif command == "stop":
            self.send_stop()
        elif command == "left":
            self.turn_left()
        elif command == "right":
            self.turn_right()
        elif command == "back":
            self.turn_back()
        elif command == "move_tiles":
            self.move_tiles(repetitions=10)

        else:
            self.get_logger().warn(f"Unknown command: {command}")
            self.send_stop()

    def move_tiles_callback(self, msg: Int32):
        tile_count = msg.data
        self.get_logger().info(f"Moving forward {tile_count} tiles.")
        self.move_tiles(repetitions=tile_count)

    def direction_to_heading(self, diff):
        # Map vector diff to a new heading
        if diff == (0, 1):
            return "north"
        elif diff == (1, 0):
            return "east"
        elif diff == (0, -1):
            return "south"
        elif diff == (-1, 0):
            return "west"
        else:
            return None

    def next_step(self):

        if self.step_count >= len(self.path) - 1:
            self.get_logger().info("Reached Goal Point")
            return

        direction_map = {
            "north": (0, 1),
            "east": (1, 0),
            "south": (0, -1),
            "west": (-1, 0),
        }

        heading_order = ["north", "east", "south", "west"]

        current = self.path[self.step_count]
        next_point = self.path[self.step_count + 1]
        diff = (next_point[0] - current[0], next_point[1] - current[1])
        new_heading = self.direction_to_heading(diff)

        if new_heading is None:
            self.get_logger().warn(f"Invalid direction from {current} to {next_point}")

        # Determine how to rotate
        current_index = heading_order.index(self.heading)
        new_index = heading_order.index(new_heading)
        turn_steps = (new_index - current_index) % 4
        self.get_logger().info(
            f"Headings: current - {self.heading}, next - {new_heading}, turn - {turn_steps}"
        )

        self.heading = new_heading
        self.step_count += 1

        if turn_steps == 1:
            self.turn_left()
            return
        elif turn_steps == 2:
            self.turn_back()
            return
        elif turn_steps == 3:
            self.turn_right()
            return
        else:
            self.move_tiles(repetitions=1)
            self.next_step()
            return

    def path_callback(self, msg: Path):
        self.path = [(index.x, index.y) for index in msg.indices]
        self.step_count = 0
        self.heading = "north"

        self.get_logger().info(f"Taking optimal path.")
        self.next_step()

    def send_cmd(self, linear=0.0, angular=0.0):
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.cmd_publisher.publish(twist)

    def send_stop(self):
        self.send_cmd(0.0, 0.0)

    def send_forward(self):
        self.send_cmd(linear=self.FORWARD_SPEED, degree=0.0)
        time.sleep(3)
        self.send_stop()

    def send_forward_speed(self, speed):
        self.send_cmd(linear=speed, degree=0.0)

    def turn_left(self):
        self.start_orientation = self.robot_orientation
        self.goal_rotation = math.radians(90)
        self.send_cmd(0.0, math.radians(-90))

        # Start a timer to periodically check the orientation
        self.timer = self.create_timer(0.02, self._check_turn_done)

    def turn_right(self):
        self.start_orientation = self.robot_orientation
        self.goal_rotation = math.radians(90)
        self.send_cmd(0.0, math.radians(90))

        # Start a timer to periodically check the orientation
        self.timer = self.create_timer(0.02, self._check_turn_done)

    def _check_turn_done(self):
        if (
            abs(self.angle_diff(self.robot_orientation, self.start_orientation))
            >= self.goal_rotation
        ):
            self.send_stop()
            self.timer.cancel()
            self.move_tiles(repetitions=1)
            self.next_step()

    def angle_diff(self, a, b):
        """Return the smallest signed difference between two angles in radians."""
        return (a - b + math.pi) % (2 * math.pi) - math.pi

    def turn_back(self):
        self.start_orientation = self.robot_orientation
        self.goal_rotation = math.radians(180)
        self.send_cmd(0.0, math.radians(90))

        # Start a timer to periodically check the orientation
        self.timer = self.create_timer(0.02, self._check_turn_done)

    def move_tiles(self, distance_per_tile=0.01, repetitions=10):

        duration = (
            distance_per_tile / self.FORWARD_SPEED
        )  # m/s  # time to move one tile

        for _ in range(repetitions):
            self.send_cmd(linear=self.FORWARD_SPEED, angular=0.0)
            time.sleep(duration)
            self.send_stop()
            # time.sleep(0.2)  # short pause between moves


def main(args=None):
    rclpy.init(args=args)

    node = MovementController()
    rclpy.spin(node)
    node.send_stop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
