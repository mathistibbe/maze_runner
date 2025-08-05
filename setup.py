from setuptools import find_packages, setup

package_name = "maze_runner"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/maze_runner.launch.py"]),
        ("share/" + package_name + "/launch", ["launch/robot.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="pi",
    maintainer_email="pi@todo.todo",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "tag_detector = maze_runner.tag_detector:main",
            "movement_controller = maze_runner.movement_controller:main",
            "movement_planner = maze_runner.movement_planner:main",
            "pathfinding = maze_runner.pathfinding:main",
            "obstacle_detector = maze_runner.obstacle_detector:main",
            "gotopoint_action_server = maze_runner.gotopoint_action_server:main",
        ],
    },
)
