import rclpy
from rclpy.node import Node
import time
from std_msgs.msg import String, Bool, Empty


class ObstacleDetector(Node):

    def __init__(self):
        super().__init__("ObstacleDetector")
        self.color_sub = self.create_subscription(
            String, "color", self.color_callback, 10
        )
        self.color_sub
        self.obstacle_pub = self.create_publisher(Empty, "obstacle_detected", 10)
