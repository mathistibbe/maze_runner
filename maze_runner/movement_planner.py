import rclpy
from rclpy.node import Node
import time
from maze_interfaces.msg import Path, Index
from std_msgs.msg import String, Bool, Int32
from aruco_interfaces.msg import ArucoMarkers
from geometry_msgs.msg import Pose, Point
from nav_msgs.msg import OccupancyGrid
from .map_creation import create_mapping, MazeMap
from .pathfinding import visualize_path, a_star
import math
import numpy as np
from rclpy.qos import qos_profile_sensor_data

from maze_interfaces.action import GoToPoint
from rclpy.action import ActionClient


class MovementPlanner(Node):
    def __init__(self):
        super().__init__("movement_planner")
        self.goal_sub = self.create_subscription(Pose, "goal_pose", self.goal_cb, 10)
        self.robot_marker_sub = self.create_subscription(
            Pose, "robot_pose", self.robot_marker_cb, 10
        )
        self._client = ActionClient(self, GoToPoint, "go_to_point")
        self.obstacle_marker_sub = self.create_subscription(
            ArucoMarkers, "obstacles", self.obstacle_marker_cb, 10
        )
        self.grid_sub = self.create_subscription(
            OccupancyGrid, "/occupancy/map/grid", self.grid_cb, qos_profile_sensor_data
        )
        self.path_pub = self.create_publisher(Path, "follow_path", 10)
        self.movement_pub = self.create_publisher(String, "execute_movement", 10)
        self.tile_pub = self.create_publisher(
            Int32, "movement_tiles", 10
        )  ##publisher for tiles
        self.goal_pos = None
        self.robot_pos = None
        self.robot_orientation = None
        self.obstacle_positions = []
        self.maze_map = None
        self.robot_index = None
        timer_period = 5  # seconds
        self.timer = self.create_timer(timer_period, self.test_grid)

    def _get_yaw_from_quaternion(self, q):
        x, y, z, w = q.x, q.y, q.z, q.w
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def position_to_grid_coordinates(self, position, grid_map):
        x = int(
            (position.x - grid_map.info.origin.position.x) / grid_map.info.resolution
        )
        y = int(
            (position.y - grid_map.info.origin.position.y) / grid_map.info.resolution
        )
        return x, y

    def grid_coordinates_to_position(self, x, y, grid_map):
        position = Point()
        position.x = (
            grid_map.info.origin.position.x
            + x * grid_map.info.resolution
            + grid_map.info.resolution / 2
        )
        position.y = (
            grid_map.info.origin.position.y
            + y * grid_map.info.resolution
            + grid_map.info.resolution / 2
        )
        position.z = 0.0
        return position

    def grid_cb(self, msg):
        if self.maze_map is None:
            self.get_logger().info("In grid_cb")
            self.destroy_subscription(self.grid_sub)
                        
            self.get_logger().info("Destroyed Subscription")

            self.maze_map = MazeMap(msg)
            #np.savetxt("maze.out", self.maze_map.map)
            shape = np.shape(self.maze_map.map)
            numbers = np.unique(self.maze_map.map)
            self.get_logger().info(
                f"---Map created---\n- resolution: {self.maze_map.resolution}\n- height: {self.maze_map.height}\n- width: {self.maze_map.width}\n- map_shape: {shape}\n- numbers: {numbers}"
            )

    def goal_cb(self, msg):
        self.goal_pos = msg.position

    def robot_marker_cb(self, msg):
        self.robot_pos = msg.position
        self.robot_orientation = msg.orientation
        #rotation = math.degrees(self._get_yaw_from_quaternion(msg.orientation))
        #self.get_logger().info(f"Rotation: {rotation}°")

    def obstacle_marker_cb(self, msg):
        self.obstacle_positions = [marker.pose.position for marker in msg.markers]

    def test_static_path(self):
        """Test the movement controll path following by sending a predefined path"""
        path = [(x, 0) for x in range(10)]
        path = path + [(9, y) for y in range(1, 10)]
        path = path + [(x, 9) for x in range(10, 20)]
        self.get_logger().info(str(path))
        self.timer.cancel()
        self.path_pub.publish(self.generate_path_msg(path))

    def feedback_callback(self, feedback_msg):
        distance = feedback_msg.feedback.distance_remaining
        self.get_logger().info(f"Distance remaining: {distance:.2f}")

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected")
            return

        self.get_logger().info("Goal accepted")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(
            f"Result: success={result.success}, message={result.message}"
        )

    def test_grid(self):
        """Tests the visualization of Maze Maps that are created from occupancy grids."""
        if self.maze_map is not None:
            self.timer.cancel()
            start = (20, 20)
            goal = (200, 200)

            path = a_star(maze=self.maze_map.map, start=start, goal=goal)
            self.get_logger().info("Computed Shortest Path")
            # visualize_path(self.maze_map.map, path, start, goal)

            #path_msg = self.generate_path_msg(path)
            #self.path_pub.publish(path_msg)

            # Convert grid to world coordinates
            # goal_point = self.grid_coordinates_to_position(
            #    goal[0], goal[1], self.maze_map.grid_msg
            # )

            # Send goal to action server
            if not self._client.wait_for_server(timeout_sec=3.0):
                self.get_logger().error("Action server not available.")
                return

            goal_msg = GoToPoint.Goal()
            goal_msg.x = 1.0
            goal_msg.y = 1.0

            self.get_logger().info(
                f"Sending action goal to: ({goal_msg.x:.2f}, {goal_msg.y:.2f})"
            )

            future = self._client.send_goal_async(
                goal_msg, feedback_callback=self.feedback_callback
            )
            future.add_done_callback(self.goal_response_callback)

    def generate_path_msg(self, path):
        indices = []
        for x, y in path:
            index_msg = Index()
            index_msg.x = x
            index_msg.y = y
            indices.append(index_msg)
        path_msg = Path()
        path_msg.indices = indices
        return path_msg

    def update_map(self):

        robot_idx, goal_idx, maze_map = create_mapping(
            self.robot_pos, self.goal_pos, self.obstacle_positions
        )

        self.maze_dict = {"goal": goal_idx, "robot": robot_idx, "map": maze_map}


def main(args=None):
    rclpy.init(args=args)
    node = MovementPlanner()
    rclpy.spin(node)
    node.send_stop()
    node.destroy_node()
    rclpy.shutdown()


##remember to add path following
