from launch_ros.substitutions import FindPackageShare

from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():

    robot_ns = LaunchConfiguration("robot_ns")
    left_motor_port = LaunchConfiguration("left_motor_port")
    right_motor_port = LaunchConfiguration("right_motor_port")
    wheel_radius = LaunchConfiguration("wheel_radius")
    base_distance = LaunchConfiguration("base_distance")

    robot_ns_launch_arg = DeclareLaunchArgument("robot_ns", default_value="rp8")
    left_motor_port_launch_arg = DeclareLaunchArgument(
        "left_motor_port", default_value="C"
    )
    right_motor_port_launch_arg = DeclareLaunchArgument(
        "right_motor_port", default_value="B"
    )
    wheel_radius_launch_arg = DeclareLaunchArgument(
        "wheel_radius", default_value="0.0275"
    )
    base_distance_launch_arg = DeclareLaunchArgument(
        "base_distance", default_value="0.116"
    )

    driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [FindPackageShare("ros2_brickpi3"), "launch", "drive.launch.py"]
                )
            ]
        ),
        launch_arguments={
            "robot_ns": LaunchConfiguration("robot_ns"),
            "left_motor_port": LaunchConfiguration("left_motor_port"),
            "right_motor_port": LaunchConfiguration("right_motor_port"),
            "wheel_radius": LaunchConfiguration("wheel_radius"),
            "base_distance": LaunchConfiguration("base_distance"),
        }.items(),
    )
    gyro = Node(
        package="ros2_brickpi3",
        namespace=robot_ns,
        executable="gyro",
        name="gyro_sensor",
        parameters=[{"port": 2}],
    )

    return LaunchDescription(
        [
            robot_ns_launch_arg,
            right_motor_port_launch_arg,
            wheel_radius_launch_arg,
            base_distance_launch_arg,
            left_motor_port_launch_arg,
            driver,
            gyro
        ]
    )
