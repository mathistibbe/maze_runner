import rclpy
from rclpy.node import Node
import time
from maze_interfaces.msg import Path, Index
from std_msgs.msg import String, Bool, Int32
from aruco_interfaces.msg import ArucoMarkers
from geometry_msgs.msg import Pose, Point
from nav_msgs.msg import OccupancyGrid
from .map_creation import create_mapping, MazeMap, visualize_weighted_map
from .pathfinding import visualize_path, a_star, b_star, a_star_weighted, reduce_path_to_straights, path_to_directions, a_star_straight_line
import math
import numpy as np
from rclpy.qos import qos_profile_sensor_data
import cv2

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
        self.clean_robot_pose_pub = self.create_publisher(
            Point, "clean_robot_pose", 10
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
        self.path = None
        self.homography_matrix = None
        self.steps = 0
        timer_period = 2  # seconds
        self.timer = self.create_timer(timer_period, self.test_grid)


    def get_homography_matrix(self):
        robot_pos = np.array([
            [0,0], # top left
            [273,-3], # top right
            [270, 183], # bottom right
            [27, 186] # bottom left
        ])
        map_points = np.array([
            [0, 0],  # top left
            [self.maze_map.width, 0],  # top right
            [self.maze_map.width, self.maze_map.height],  # bottom right
            [0, self.maze_map.height]   # bottom left
        ])
        self.homography_matrix, _ = cv2.findHomography(robot_pos, map_points)
    


    def position_to_grid_coordinates(self, position, maze_map):
        """Convert a position to grid coordinates based on the map's origin and resolution.
        Args:
            position (Point): The position in the world frame.
            grid_map (OccupancyGrid): The occupancy grid map.
        Returns:
            tuple: The grid coordinates (x, y)."""
        x = int(
            (abs(position.x) - maze_map.origin.position.x) / maze_map.resolution
        )
        y = int(
            (abs(position.y) - maze_map.origin.position.y) / maze_map.resolution
        )

        # apply homography transformation
        if self.homography_matrix is not None:
            point = np.array([[x], [y], [1]])
            transformed_point = self.homography_matrix @ point
            transformed_point /= transformed_point[2]
            x, y = int(transformed_point[0]), int(transformed_point[1])

        return x, y

    def apply_homography_position(self, pose):
        """Apply homography transformation to a Point."""
        point_array = np.array([[pose.x * 100], [pose.y * 100], [1]])
        transformed_point = self.homography_matrix @ point_array
        transformed_point /= transformed_point[2]
        msg = Point()
        msg.x = float(transformed_point[0, 0] / 100)
        msg.y = float(transformed_point[1, 0] / 100)
        msg.z = 0.0
        return msg

    def grid_coordinates_to_position(self, x, y, maze_map):
        """ Convert grid coordinates to a position in the world frame.
        Args:
            x (int): The x coordinate in the grid.
            y (int): The y coordinate in the grid.
            grid_map (OccupancyGrid): The occupancy grid map.
        Returns:
            Point: The position in the world frame.
        """
        position = Point()
        position.x = (
            maze_map.origin.position.x
            + x * maze_map.resolution
            + maze_map.resolution / 2
        )
        position.y = (
            maze_map.origin.position.y
            + y * maze_map.resolution
            + maze_map.resolution / 2
        )
        position.z = 0.0
        return position

    def grid_cb(self, msg):
        """Callback for the occupancy grid map."""
        if self.maze_map is None:
            self.get_logger().info("In grid_cb")
            self.destroy_subscription(self.grid_sub)
                        
            self.get_logger().info("Destroyed Subscription")

            self.maze_map = MazeMap(msg)
            #np.savetxt("maze.out", self.maze_map.map)
            shape = np.shape(self.maze_map.map)
            numbers = np.unique(self.maze_map.map)
            self.get_homography_matrix()
            self.get_logger().info(
                f"---Map created---\n- resolution: {self.maze_map.resolution}\n- height: {self.maze_map.height}\n- width: {self.maze_map.width}\n- map_shape: {shape}\n- numbers: {numbers}"
            )

    def goal_cb(self, msg):
        self.goal_pos = msg.position

    def robot_marker_cb(self, msg):
        self.robot_pos = msg.position
        self.robot_orientation = msg.orientation
        if self.homography_matrix is not None:
            robot_coord = self.apply_homography_position(msg.position)
            self.clean_robot_pose_pub.publish(robot_coord)
        #rotation = math.degrees(self._get_yaw_from_quaternion(msg.orientation))
        #self.get_logger().info(f"Rotation: {rotation}°")

    def obstacle_marker_cb(self, msg):
        self.obstacle_positions = [marker.pose.position for marker in msg.markers]

    def feedback_callback(self, feedback_msg):
        distance = feedback_msg.feedback.distance_remaining
        #self.get_logger().info(f"Distance remaining: {distance:.2f}")

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
        if result.success:
            if self.steps >= len(self.path):
                self.get_logger().info("Reached the goal successfully!")
            else:
                self.send_goal_to_action_server()

    def test_grid(self):
        """Tests the visualization of Maze Maps that are created from occupancy grids."""
        if self.maze_map is not None and self.robot_pos is not None and self.goal_pos is not None:
            self.timer.cancel()
            start = self.position_to_grid_coordinates(self.robot_pos, self.maze_map)
            goal = self.position_to_grid_coordinates(self.goal_pos, self.maze_map)

            #start = (30,60)
            #goal = (230, 200)

            self.get_logger().info(f"Start: {start}, Goal: {goal}")

            #path = a_star_straight_line(
            #    self.maze_map.weighted_cost_map, start, goal
            #)
            
            path = a_star_weighted(
                self.maze_map.weighted_cost_map, start, goal
            )

            self.maze_map.visualize_cost_map(path=path)
            reduced_path = reduce_path_to_straights(path)

            directions = path_to_directions(reduced_path)
            self.get_logger().info("Computed Shortest Path")
            self.get_logger().info(f"Path: {directions}")
            self.get_logger().info(f"Reduced Path: {reduced_path}")
            # visualize_path(self.maze_map.map, reduced_path, start, goal)


            # Send goal to action server


            if not self._client.wait_for_server(timeout_sec=3.0):
                self.get_logger().error("Action server not available.")
                return
            
            self.path = directions
            self.steps = 0
            self.send_goal_to_action_server()
            
    def test_map_transformation(self):
        if self.maze_map is not None:
            self.timer.cancel()
            self.get_logger().info("Testing map transformation")
            self.maze_map.visualize_cost_map()


                
            

    def send_goal_to_action_server(self):
        """Sends the goal to the action server."""
        #goal_point = self.grid_coordinates_to_position(self.path[self.steps][0], self.path[self.steps][1], self.maze_map)
        
        goal_msg = GoToPoint.Goal()
        goal_msg.x = float(self.path[self.steps][0])
        goal_msg.y = float(self.path[self.steps][1])
        self.steps += 1
        self.get_logger().info(f"Sending goal: {goal_msg.x}, {goal_msg.y}")
        self._client.wait_for_server()
        send_goal_future = self._client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

            

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
