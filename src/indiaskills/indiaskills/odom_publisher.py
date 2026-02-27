import rclpy
from rclpy.node import Node
import math
from geometry_msgs.msg import Vector3, Quaternion
from nav_msgs.msg import Odometry

class OdometryNode(Node):
    def __init__(self):
        super().__init__('odom_publisher')
        self.wheelbase_L = 0.35
        self.dist_per_tick = (2.0 * math.pi * 0.05) / 1440.0 # Adjust 1440 based on your encoder PPR
        self.x, self.y, self.th = 0.0, 0.0, 0.0
        self.prev_left_ticks, self.prev_right_ticks = 0, 0
        self.last_time = self.get_clock().now()
        self.first_run = True

        self.create_subscription(Vector3, 'wheel_ticks', self.ticks_callback, 10)
        self.odom_pub = self.create_publisher(Odometry, 'wheel/odom', 10)

    def ticks_callback(self, msg):
        current_time = self.get_clock().now()
        if self.first_run:
            self.prev_left_ticks, self.prev_right_ticks = msg.x, msg.y
            self.last_time, self.first_run = current_time, False
            return

        dt = (current_time.nanoseconds - self.last_time.nanoseconds) / 1e9
        dist_left = (msg.x - self.prev_left_ticks) * self.dist_per_tick
        dist_right = (msg.y - self.prev_right_ticks) * self.dist_per_tick

        delta_dist = (dist_right + dist_left) / 2.0
        delta_th = (dist_right - dist_left) / self.wheelbase_L

        self.x += delta_dist * math.cos(self.th + (delta_th / 2.0))
        self.y += delta_dist * math.sin(self.th + (delta_th / 2.0))
        self.th += delta_th

        odom = Odometry()
        odom.header.stamp = current_time.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.twist.twist.linear.x = delta_dist / dt if dt > 0 else 0.0
        odom.twist.twist.angular.z = delta_th / dt if dt > 0 else 0.0
        self.odom_pub.publish(odom)

        self.prev_left_ticks, self.prev_right_ticks, self.last_time = msg.x, msg.y, current_time

def main():
    rclpy.init()
    node = OdometryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()