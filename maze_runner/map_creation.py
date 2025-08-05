import numpy as np
import math
from geometry_msgs.msg import Pose, Point
from nav_msgs.msg import OccupancyGrid, MapMetaData
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt


class MazeMap:

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid_msg = occupancy_grid
        self.origin = occupancy_grid.info.origin
        self.creation_time = occupancy_grid.info.map_load_time
        self.resolution = occupancy_grid.info.resolution
        self.width = occupancy_grid.info.width
        self.height = occupancy_grid.info.height
        self.map = np.reshape(occupancy_grid.data, (self.height, self.width)).T
        self.map[self.map > 0] = 0
        self.map[self.map < 0] = 1
        self.map = self.apply_custom_convolution(self.map)
        self.downscale_map(factor = 10)
        self.weighted_cost_map = compute_weighted_cost_map(self.map, alpha=0.5)
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

    def downscale_map(self, factor: int) -> np.ndarray:
        """
        Downscale a 2D array by an integer factor.
        Pads the array if needed so dimensions are divisible by the factor.

        Args:
            factor (int): Downscaling factor.

        Returns:
            np.ndarray: Downscaled occupancy grid.
        """


        rows, cols = self.map.shape
        pad_rows = (factor - rows % factor) % factor
        pad_cols = (factor - cols % factor) % factor

        # Pad with zeros (assuming padding is free space)
        padded = np.pad(self.map, ((0, pad_rows), (0, pad_cols)), mode='constant', constant_values=0)

        new_rows = padded.shape[0] // factor
        new_cols = padded.shape[1] // factor

        # Reshape and max-pool
        reshaped = padded.reshape(new_rows, factor, new_cols, factor)
        downscaled = reshaped.max(axis=(1, 3))
        self.resolution *= factor
        self.map = downscaled

    def visualize_cost_map(self, path=None):
        visualize_weighted_map(cost_map=self.weighted_cost_map, original_maze=self.map, path=path)

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

def compute_weighted_cost_map(maze: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Compute a cost map for A* that favors being far from walls.

    Parameters:
    - maze: 2D numpy array, where 0 = free, 1 = wall.
    - alpha: float, how strongly to bias toward open space (higher = more centered paths).

    Returns:
    - cost_map: 2D numpy array of same shape as maze, with higher cost near walls.
    """
    # Step 1: Invert maze so walls = 0, free = 1
    free_space = (maze == 0).astype(np.uint8)
    
    # Step 2: Compute distance to the nearest wall
    distance_from_wall = distance_transform_edt(free_space)

    # Step 3: Normalize distance map
    max_dist = np.max(distance_from_wall)
    if max_dist > 0:
        normalized = distance_from_wall / max_dist
    else:
        normalized = distance_from_wall

    # Step 4: Invert and weight (walls = high cost, centers = low cost)
    # You can use (1 - normalized) to turn "distance from wall" into "penalty"
    cost_map = 1.0 + alpha * (1.0 - normalized)  # Add 1 so cost is never 0

    # Optional: Set walls to inf cost
    cost_map[maze == 1] = np.inf

    return cost_map

import matplotlib.pyplot as plt

def visualize_weighted_map(cost_map, original_maze=None, path=None, filename='./src/weighted_cost_map.png'):
    plt.figure(figsize=(8, 8))
    plt.imshow(cost_map, cmap='viridis', origin='lower')
    plt.colorbar(label='Cost')

    if original_maze is not None:
        wall_y, wall_x = np.where(original_maze == 1)
        plt.scatter(wall_x, wall_y, color='red', s=1, label='Walls')

    if path is not None:
        path = np.array(path)
        plt.plot(path[:, 1], path[:, 0], color='white', linewidth=2, label='Path')

    plt.legend()
    plt.title("Weighted Cost Map Visualization")
    plt.xlabel("X (columns)")
    plt.ylabel("Y (rows)")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()