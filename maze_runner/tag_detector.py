##Node for processing data from /aruco/transformed
## frame_id: zed_camera_frame
## markers[]


##personalRobotID = 4 (Our Robot)
##fieldtagID = 15

##/arucotransformed data:
## ID
## pose.position: Vector3.(x,y,z)
## pose.orientation: quarternion [x,y,z,w]

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from aruco_interfaces.msg import ArucoMarkers, ArucoMarker
from geometry_msgs.msg import Pose, PoseStamped


class TagDetector(Node):
    def __init__(self):
        super().__init__("tag_detector")

        self.robot_id = 4  # Our Robot
        self.goal_id = 15  # Goal tag
        self.corner_id = 13 # Corner tag

        # Publishers for robot, goal, and obstacles
        self.robot_pub = self.create_publisher(Pose, "robot_pose", 10)
        self.goal_pub = self.create_publisher(Pose, "goal_pose", 10)
        self.obstacle_pub = self.create_publisher(ArucoMarkers, "obstacles", 10)
        #self.rviz_robot_pub = self.create_publisher(PoseStamped, "rviz_robot_pose", 10)
        #self.rviz_goal_pub = self.create_publisher(PoseStamped, "rviz_goal_pose", 10)

        # Subscriber for the aruco marker detections
        self.subscription = self.create_subscription(
            ArucoMarkers, "/aruco/markers/uncertain", self.aruco_callback, qos_profile_sensor_data
        )

        self.get_logger().info(
            "Tag detector node started. Listening to /aruco/markers/uncertain"
        )

    def aruco_callback(self, msg: ArucoMarkers):
        robot_marker = None
        goal_marker = None
        obstacles = []

        for marker in msg.markers:
            if marker.id == self.robot_id:
                robot_marker = marker
            elif marker.id == self.goal_id:
                goal_marker = marker
            else:
                obstacles.append(marker)

        # Publish robot pose if detected

        if robot_marker is not None:
            flipped_pose = Pose()
           # flipped_pose.position.x = robot_marker.pose.position.y
           # flipped_pose.position.y = robot_marker.pose.position.x 
           # flipped_pose.position.z = robot_marker.pose.position.z
           # flipped_pose.orientation = robot_marker.pose.orientation
            robotrvpose = PoseStamped()
            robotrvpose.header.frame_id = 'map'
            robotrvpose.pose = robot_marker.pose
            robot_marker.pose.position.z = 0.0
            robot_marker.pose.orientation.x = 0.0
            robot_marker.pose.orientation.y = 0.0
            #self.rviz_robot_pub.publish(robotrvpose)
            self.robot_pub.publish(robot_marker.pose)
            
            

        # Publish goal pose if detected
        if goal_marker is not None:
            flipped_gpose = Pose()
           # flipped_gpose.position.x = goal_marker.pose.position.y
           # flipped_gpose.position.y = goal_marker.pose.position.x 
           # flipped_gpose.position.z = goal_marker.pose.position.z
           # flipped_gpose.orientation = goal_marker.pose.orientation
            goalrvpose = PoseStamped()
            goalrvpose.header.frame_id = 'map'
            goalrvpose.pose = goal_marker.pose
            goal_marker.pose.position.z = 0.0
            goal_marker.pose.orientation.x = 0.0
            goal_marker.pose.orientation.y = 0.0
            #self.rviz_goal_pub.publish(goalrvpose)
            self.goal_pub.publish(goal_marker.pose)

        # Publish obstacles if any
        if obstacles:
            obstacles_msg = ArucoMarkers()
            obstacles_msg.header = msg.header
            obstacles_msg.markers = obstacles
            self.obstacle_pub.publish(obstacles_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TagDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
