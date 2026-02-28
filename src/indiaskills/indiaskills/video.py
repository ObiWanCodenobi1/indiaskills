import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np

class VisionInspector(Node):
    def __init__(self):
        super().__init__('vision_inspector')
        
        # Subscribe to the camera feed
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10)
            
        # Publisher to tell the state machine if the object is defective
        self.defect_pub = self.create_publisher(Bool, '/is_defective', 10)
        
        self.bridge = CvBridge()
        self.get_logger().info("Vision Inspector Node Started. Waiting for video...")

        # Define HSV range for the "Good" object (Example: Blue)
        # YOU MUST TUNE THESE VALUES AT THE COMPETITION HALL!
        self.lower_color = np.array([100, 150, 50])
        self.upper_color = np.array([140, 255, 255])

        # Minimum pixel area to be considered a valid object
        self.min_area = 5000 

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # 1. Convert to HSV color space
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # 2. Create a mask to isolate the target color
        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)

        # 3. Clean up the mask (remove noise)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 4. Find contours (shapes) in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        is_defective = True # Assume defective until proven good

        if len(contours) > 0:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            # If the object is the right color and large enough, it's good!
            if area > self.min_area:
                is_defective = False
                
                # Optional: Draw a green box around it for debugging
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(cv_image, "GOOD", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 5. Publish the result
        defect_msg = Bool()
        defect_msg.data = is_defective
        self.defect_pub.publish(defect_msg)

        # Optional: Show the camera feed on the Pi desktop for debugging
        # cv2.imshow("Inspection Camera", cv_image)
        # cv2.imshow("Mask", mask)
        # cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = VisionInspector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()