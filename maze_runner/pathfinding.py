import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib import colors


def visualize_path(maze, path, start, goal, filename="path_visualization"):

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
    bounds = [0, 1, 2, 3, 4, 5]
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


def a_star(maze: np.ndarray, start: tuple, goal: tuple):
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

        step_size = 10  # Number of cells per move

        for dx, dy in [
            (-step_size, 0),
            (step_size, 0),
            (0, -step_size),
            (0, step_size),
        ]:
            neighbor = (current[0] + dx, current[1] + dy)

            if (
                0 <= neighbor[0] < rows
                and 0 <= neighbor[1] < cols
                and maze[neighbor[0], neighbor[1]] == 0
            ):
                # Optionally: Check all intermediate cells in the stride for safety
                path_clear = True
                for i in range(1, step_size):
                    intermediate = (
                        current[0] + dx // step_size * i,
                        current[1] + dy // step_size * i,
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

    return None


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
