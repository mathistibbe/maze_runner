
import numpy as np
import heapq
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


def a_star(maze: np.ndarray, start: tuple, goal: tuple, step_size=1):
    rows, cols = maze.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        directions = [
            (-step_size, 0),
            (step_size, 0),
            (0, -step_size),
            (0, step_size),
        ]

        # Standard moves (fixed step_size)
        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if (
                0 <= neighbor[0] < rows
                and 0 <= neighbor[1] < cols
                and maze[neighbor[0], neighbor[1]] == 0
            ):
                path_clear = True
                for i in range(1, step_size + 1):
                    intermediate = (
                        current[0] + (dx // step_size) * i if dx != 0 else current[0],
                        current[1] + (dy // step_size) * i if dy != 0 else current[1],
                    )
                    if (
                        intermediate[0] < 0 or intermediate[0] >= rows or
                        intermediate[1] < 0 or intermediate[1] >= cols or
                        maze[intermediate[0], intermediate[1]] != 0
                    ):
                        path_clear = False
                        break
                if not path_clear:
                    continue

                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

        # Lenient final move: allow any straight move to goal if path is clear
        if current[0] == goal[0]:
            step = 1 if goal[1] > current[1] else -1
            dist = abs(goal[1] - current[1])
            path_clear = True
            for i in range(1, dist + 1):
                intermediate = (current[0], current[1] + step * i)
                if (
                    intermediate[1] < 0 or intermediate[1] >= cols or
                    maze[intermediate[0], intermediate[1]] != 0
                ):
                    path_clear = False
                    break
            if path_clear:
                neighbor = goal
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
        elif current[1] == goal[1]:
            step = 1 if goal[0] > current[0] else -1
            dist = abs(goal[0] - current[0])
            path_clear = True
            for i in range(1, dist + 1):
                intermediate = (current[0] + step * i, current[1])
                if (
                    intermediate[0] < 0 or intermediate[0] >= rows or
                    maze[intermediate[0], intermediate[1]] != 0
                ):
                    path_clear = False
                    break
            if path_clear:
                neighbor = goal
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

    return None

def b_star(maze: np.ndarray, start: tuple, goal: tuple, step_size=1):
    rows, cols = maze.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

    while open_set:
        _, current = heapq.heappop(open_set)

        # If goal is within step_size, add goal and return path
        if abs(current[0] - goal[0]) + abs(current[1] - goal[1]) <= step_size:
            path = [goal, current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        directions = [
            (-step_size, 0),
            (step_size, 0),
            (0, -step_size),
            (0, step_size),
        ]

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if (
                0 <= neighbor[0] < rows
                and 0 <= neighbor[1] < cols
                and maze[neighbor[0], neighbor[1]] == 0
            ):
                # Check all intermediate cells between current and neighbor
                path_clear = True
                for i in range(1, step_size + 1):
                    intermediate = (
                        current[0] + (dx // step_size) * i if dx != 0 else current[0],
                        current[1] + (dy // step_size) * i if dy != 0 else current[1],
                    )
                    if (
                        intermediate[0] < 0 or intermediate[0] >= rows or
                        intermediate[1] < 0 or intermediate[1] >= cols or
                        maze[intermediate[0], intermediate[1]] != 0
                    ):
                        path_clear = False
                        break
                if not path_clear:
                    continue

                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

    return None

def c_star(maze: np.ndarray, start: tuple, goal: tuple):
    rows, cols = maze.shape
    step_size = 10
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        # Try moving in steps of 10 cells
        for dx, dy in [(-step_size, 0), (step_size, 0), (0, -step_size), (0, step_size)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                # Check all intermediate cells for obstacles
                path_clear = True
                for i in range(1, step_size + 1):
                    intermediate = (
                        current[0] + (dx // step_size) * i if dx != 0 else current[0],
                        current[1] + (dy // step_size) * i if dy != 0 else current[1],
                    )
                    if maze[intermediate[0], intermediate[1]] != 0:
                        path_clear = False
                        break
                if not path_clear:
                    continue
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

        # Try direct move to goal if aligned and path is clear (final step)
        if current[0] == goal[0]:
            step = 1 if goal[1] > current[1] else -1
            dist = abs(goal[1] - current[1])
            path_clear = True
            for i in range(1, dist + 1):
                intermediate = (current[0], current[1] + step * i)
                if maze[intermediate[0], intermediate[1]] != 0:
                    path_clear = False
                    break
            if path_clear:
                neighbor = goal
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g
                    heapq.heappush(open_set, (f_score, neighbor))
        elif current[1] == goal[1]:
            step = 1 if goal[0] > current[0] else -1
            dist = abs(goal[0] - current[0])
            path_clear = True
            for i in range(1, dist + 1):
                intermediate = (current[0] + step * i, current[1])
                if maze[intermediate[0], intermediate[1]] != 0:
                    path_clear = False
                    break
            if path_clear:
                neighbor = goal
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g
                    heapq.heappush(open_set, (f_score, neighbor))

    return None

def reduce_path_to_straights(path):
    """
    Reduces a path of grid indices to only the points where the direction changes.
    Keeps the start, end, and all "corners".
    Args:
        path (list of (x, y)): The original path as a list of grid indices.
    Returns:
        list of (x, y): Reduced path with only waypoints at direction changes.
    """
    if not path or len(path) < 2:
        return path

    reduced = [path[0]]
    prev_dx = path[1][0] - path[0][0]
    prev_dy = path[1][1] - path[0][1]

    for i in range(2, len(path)):
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        if (dx, dy) != (prev_dx, prev_dy):
            reduced.append(path[i-1])
        prev_dx, prev_dy = dx, dy

    reduced.append(path[-1])
    return reduced

def path_to_directions(reduced_path):
    """
    Converts a reduced path to a list of (distance, direction) tuples.
    Directions: 0, 1, 2, 3
    """
    if not reduced_path or len(reduced_path) < 2:
        return []

    directions = []
    for i in range(1, len(reduced_path)):
        x0, y0 = reduced_path[i-1]
        x1, y1 = reduced_path[i]
        dx = x1 - x0
        dy = y1 - y0

        if dx > 0 and dy == 0:
            direction = 3 # W
            distance = dx
        elif dx < 0 and dy == 0:
            direction = 1 # E
            distance = -dx
        elif dy > 0 and dx == 0:
            direction = 0 # N
            distance = dy
        elif dy < 0 and dx == 0:
            direction = 2 # S
            distance = -dy
        else:
            raise ValueError(f"Non-straight segment from {reduced_path[i-1]} to {reduced_path[i]}")

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
