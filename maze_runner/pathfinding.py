
import numpy as np
import heapq
from collections import deque
import matplotlib.pyplot as plt
from matplotlib import colors


def visualize_path(maze, path, start, goal, filename="./src/path_visualization"):

    rows, cols = maze.shape
    maze = np.where(maze == -1, 0, maze)
    # Create display_maze initialized as all unknown (gray = value 5)
    display_maze = np.copy(maze)  # 5 = unknown (we'll define it in colormap)

    # Set known traversable and obstacle cells
    display_maze[maze == 0] = 0  # free → white
    display_maze[maze == 1] = 1  # occupied → black
    display_maze[maze == -1] = 5  # unknown → gray (already default, but explicit)

    # Mark the path
    for x, y in path:
        display_maze[x, y] = 2  # path → blue

    # Mark start and goal
    sx, sy = start
    gx, gy = goal
    display_maze[sx, sy] = 3  # start → green
    display_maze[gx, gy] = 4  # goal → red

    cmap = colors.ListedColormap(
        [
            "white",  # 0 - free
            "black",  # 1 - obstacle
            "#3399FF",  # 2 - path
            "green",  # 3 - start
            "red",  # 4 - goal
            "gray",
        ]
    )
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots()
    ax.imshow(display_maze, cmap=cmap, norm=norm)

    # Draw gridlines
    #  ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    #  ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    #  ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    #  ax.tick_params(which="minor", bottom=False, left=False)

    # Hide tick labels
    ax.set_xticks([])
    ax.set_yticks([])

    # Adjust size & save
    fig.set_size_inches((8, 8))
    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=500)
    print(f"Path visualization saved as '{filename}.png'")
    plt.close()

def a_star_weighted_v2(cost_map, start, goal, step=1):
    """
    A* search on a weighted 2D cost map. If the start is inside an obstacle, 
    finds the closest free cell and starts from there.

    Parameters:
    - cost_map: 2D numpy array with float values. `np.inf` = blocked.
    - start, goal: (y, x) tuples.
    - step: int > 0, how many cells to move per step.

    Returns:
    - path: list of (y, x) tuples from start to goal, or None if no path found.
    """
    H, W = cost_map.shape
    sy, sx = start
    gy, gx = goal

    def heuristic(a, b):
        return np.hypot(b[0] - a[0], b[1] - a[1])

    def get_neighbors(pos):
        y, x = pos
        offsets = [(-step, 0), (step, 0), (0, -step), (0, step),
                   (-step, -step), (-step, step), (step, -step), (step, step)]
        for dy, dx in offsets:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                if cost_map[ny, nx] != np.inf:
                    yield (ny, nx)

    def find_closest_free_cell(start):
        """Breadth-first search to find the nearest non-obstacle cell."""
        visited = set()
        queue = deque([start])
        while queue:
            cy, cx = queue.popleft()
            if (cy, cx) in visited:
                continue
            visited.add((cy, cx))
            if 0 <= cy < H and 0 <= cx < W and cost_map[cy, cx] != np.inf:
                return (cy, cx)
            # Check 4-connected neighbors (more robust against narrow passages)
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < H and 0 <= nx < W and (ny, nx) not in visited:
                    queue.append((ny, nx))
        return None

    # --- Adjust start if it's in a wall ---
    if cost_map[sy, sx] == np.inf:
        new_start = find_closest_free_cell(start)
        if new_start is None:
            return None  # No free cell found
        sy, sx = new_start
        start = (sy, sx)

    # --- A* Search ---
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start))

    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current_cost, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for neighbor in get_neighbors(current):
            tentative_g = current_cost + cost_map[neighbor]
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, tentative_g, neighbor))
                came_from[neighbor] = current

    return None  # No path found

def a_star_weighted(cost_map, start, goal, step=1):
    """
    A* search on a weighted 2D cost map.

    Parameters:
    - cost_map: 2D numpy array with float values. `np.inf` = blocked.
    - start, goal: (y, x) tuples.
    - step: int > 0, how many cells to move per step.

    Returns:
    - path: list of (y, x) tuples from start to goal, or None if no path found.
    """
    H, W = cost_map.shape
    sy, sx = start
    gy, gx = goal

    def heuristic(a, b):
        return np.hypot(b[0] - a[0], b[1] - a[1])

    def get_neighbors(pos):
        y, x = pos
        offsets = [(-step, 0), (step, 0), (0, -step), (0, step),
                   (-step, -step), (-step, step), (step, -step), (step, step)]
        for dy, dx in offsets:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                if cost_map[ny, nx] != np.inf:
                    yield (ny, nx)

    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))

    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current_cost, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for neighbor in get_neighbors(current):
            tentative_g = current_cost + cost_map[neighbor]
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, tentative_g, neighbor))
                came_from[neighbor] = current

    return None  # No path found

def a_star_straight_line(cost_map, start, goal, step=1):
    """
    A* search on a weighted 2D cost map, prioritizing straight-line paths.

    Parameters:
    - cost_map: 2D numpy array with float values. `np.inf` = blocked.
    - start, goal: (y, x) tuples.
    - step: int > 0, how many cells to move per step.

    Returns:
    - path: list of (y, x) tuples from start to goal, or None if no path found.
    """
    H, W = cost_map.shape
    sy, sx = start
    gy, gx = goal

    def heuristic(a, b):
        return np.hypot(b[0] - a[0], b[1] - a[1])  # Euclidean distance

    def get_neighbors(pos, last_direction):
        """Return valid neighbors considering the step size and previous movement direction"""
        y, x = pos
        neighbors = []
        
        if last_direction is None:  # First move, allow all directions
            offsets = [(-step, 0), (step, 0), (0, -step), (0, step)]
        else:
            # Prioritize straight-line movement (same direction as previous)
            offsets = []
            if last_direction in ['vertical', None]:
                offsets.extend([(-step, 0), (step, 0)])
            if last_direction in ['horizontal', None]:
                offsets.extend([(0, -step), (0, step)])

        for dy, dx in offsets:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                if cost_map[ny, nx] != np.inf:
                    neighbors.append((ny, nx))

        return neighbors

    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start, None))

    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current_cost, current, last_direction = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]  # reverse path to start -> goal

        for neighbor in get_neighbors(current, last_direction):
            tentative_g = current_cost + cost_map[neighbor]
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, tentative_g, neighbor, 
                                         'vertical' if neighbor[0] != current[0] else 'horizontal'))
                came_from[neighbor] = current

    return None 

    

def reduce_path_to_straights(path):
    """
    Reduces a path of grid indices to only the points where the direction changes.
    Handles both cardinal and diagonal straight lines.

    Assumes the path is in (x, y) format (Cartesian coordinates).

    Args:
        path (list of (x, y)): The original path as a list of grid indices.

    Returns:
        list of (x, y): Reduced path with only waypoints at direction changes.
    """
    if not path or len(path) < 2:
        return path

    reduced = [path[0]]

    def direction(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return (dx, dy)

    prev_dir = direction(path[0], path[1])

    for i in range(2, len(path)):
        curr_dir = direction(path[i - 1], path[i])
        if curr_dir != prev_dir:
            reduced.append(path[i - 1])
        prev_dir = curr_dir

    reduced.append(path[-1])
    return reduced


def path_to_directions(reduced_path):
    """
    Converts a reduced path to a list of (distance, direction) tuples.
    Supports 8 directions:
        0: N, 1: NE, 2: E, 3: SE,
        4: S, 5: SW, 6: W, 7: NW
    Args:
        reduced_path (list of (x, y)): Path reduced to straights and diagonals.
    Returns:
        list of (distance, direction) tuples.
    """
    if not reduced_path or len(reduced_path) < 2:
        return []

    direction_map = {
        (0, 1): 0,    # N
        (1, 1): 1,    # NW
        (1, 0): 2,    # W
        (1, -1): 3,   # SW
        (0, -1): 4,   # S
        (-1, -1): 5,  # SE
        (-1, 0): 6,   # E
        (-1, 1): 7    # NE
    }

    directions = []

    for i in range(1, len(reduced_path)):
        x0, y0 = reduced_path[i-1]
        x1, y1 = reduced_path[i]
        dx = x1 - x0
        dy = y1 - y0

        # Normalize direction vector to unit step
        step_dx = int(dx / max(abs(dx), abs(dy))) if dx != 0 else 0
        step_dy = int(dy / max(abs(dx), abs(dy))) if dy != 0 else 0

        direction = direction_map.get((step_dx, step_dy))
        if direction is None:
            raise ValueError(f"Unsupported direction from {reduced_path[i-1]} to {reduced_path[i]}")

        # Distance = number of steps
        distance = max(abs(dx), abs(dy))

        directions.append((distance, direction))

    return directions


# ---- TEST CASE ----

# dummy_array = np.zeros((101, 101))
# for x in range(50, 60):
#    for y in range(50, 101):
#        dummy_array[x, y] = 1  # obstacle block

# dummy_dict = {"goal": (80, 90), "robot": (45, 100), "map": dummy_array}

# path = a_star(dummy_dict["map"], dummy_dict["robot"], dummy_dict["goal"])
# if path:
#   print(f"Path found! Length: {len(path)}")
#  print("First steps:", path[:10])
# visualize_path(dummy_dict["map"], path, dummy_dict["robot"], dummy_dict["goal"])
# else:
#   print("No path found.")
