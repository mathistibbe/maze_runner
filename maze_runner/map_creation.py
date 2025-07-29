import numpy as np
import math
from geometry_msgs.msg import Pose, Point
from nav_msgs.msg import OccupancyGrid, MapMetaData


class MazeMap:

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid_msg = occupancy_grid
        self.creation_time = occupancy_grid.info.map_load_time
        self.resolution = occupancy_grid.info.resolution
        self.width = occupancy_grid.info.width
        self.height = occupancy_grid.info.height
        self.map = np.reshape(occupancy_grid.data, (self.height, self.width))
        self.map[self.map > 0] = 0
        self.map[self.map < 0] = 1
        self.map = self.apply_custom_convolution(self.map)
        # np.savetxt("maze.out",self.map)

    def apply_custom_convolution(self, arr):
        output = arr.copy()
        h, w = arr.shape
        window_size = 30

        for i in range(h - window_size + 1):
            for j in range(w - window_size + 1):
                window = arr[i : i + window_size, j : j + window_size]

                # Check if border is all zeros
                top = window[0, :]
                bottom = window[-1, :]
                left = window[:, 0]
                right = window[:, -1]

                if (
                    np.all(top == 0)
                    and np.all(bottom == 0)
                    and np.all(left == 0)
                    and np.all(right == 0)
                ):
                    output[i : i + window_size, j : j + window_size] = 0

        return output


def create_mapping(robot_point, goal_point, obstacle_points):
    """Creates a map representing a 5,05m x 5,05m plane as an np.array. Blocking obstacles are represented by a 0
    Args:
        robot_point:
        goal_point:
        obstacle_point:


    Returns:
    """
    offset = 51
    array_map = np.zeros((101, 101))
    for obstacle in obstacle_points:
        x = int(obstacle.x / 0.05) + offset
        y = int(obstacle.y / 0.05) + offset
        for x_idx in range(x - 4, x + 5):
            for y_idx in range(y - 4, y + 5):
                array_map[x_idx, y_idx] = 1

    robot_idx = (int(robot_point.x / 0.05) + offset, int(robot_point.y / 0.05) + offset)
    goal_idx = (int(goal_point.x / 0.05) + offset, int(goal_point.y / 0.05) + offset)

    return robot_idx, goal_idx, array_map
