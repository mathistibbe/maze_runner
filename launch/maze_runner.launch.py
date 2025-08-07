from launch_ros.substitutions import FindPackageShare

from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():

    robot_ns = LaunchConfiguration("robot_ns")
    initial_heading = LaunchConfiguration("initial_heading")

    robot_ns_launch_arg = DeclareLaunchArgument("robot_ns", default_value="rp8")

    initial_heading_launch_arg = DeclareLaunchArgument(
        "initial_heading", default_value="0"
    )

    # movementcontroller = Node(
    #    package="maze_runner",
    #    namespace=robot_ns,
    #    executable="movement_controller",
    #    name="movement_controller",
    #    output="screen",
    # )
    go_to_point_action_server = Node(
        package="maze_runner",
        namespace=robot_ns,
        executable="gotopoint_action_server",
        name="gotopoint_action_server",
        output="screen",
        parameters=[{"initial_heading": LaunchConfiguration("initial_heading")}],
    )
    movementplanner = Node(
        package="maze_runner",
        namespace=robot_ns,
        executable="movement_planner",
        name="movement_planner",
        output="screen",
    )
    tagdetector = Node(
        package="maze_runner",
        namespace=robot_ns,
        executable="tag_detector",
        name="tag_detector",
        output="screen",
    )

    return LaunchDescription(
        [
            initial_heading_launch_arg,
            robot_ns_launch_arg,
            go_to_point_action_server,
            # movementcontroller,
            movementplanner,
            tagdetector,
        ]
    )
