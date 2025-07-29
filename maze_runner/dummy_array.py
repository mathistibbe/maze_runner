import numpy as np


dummy_array = np.ones((101, 101))
for x in range(50, 60):
    for y in range(50, 60):
        dummy_array[x, y] = 0


dummy_dict = {"goal": (100, 100), "robot": (0, 0), "map": dummy_array}
