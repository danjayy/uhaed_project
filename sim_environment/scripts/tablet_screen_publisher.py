#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import String


class TabletScreenPublisher(Node):
    def __init__(self):
        super().__init__('tablet_screen_publisher')
        
        # Create a bridge between ROS and OpenCV
        self.bridge = CvBridge()
        
        # Publisher for the screen image
        self.image_pub = self.create_publisher(
            Image, 
            '/pepper/tablet_screen/image', 
            10
        )
        
        # Subscriber for image commands (file paths)
        self.image_command_sub = self.create_subscription(
            String,
            '/pepper/tablet_screen/display_image',
            self.display_image_callback,
            10
        )
        
        # Store current image
        self.current_image = self.create_blank_image()
        
        # Timer to continuously publish the image
        self.timer = self.create_timer(0.1, self.publish_image)  # 10 Hz
        
        self.get_logger().info('Tablet Screen Publisher initialized')
        self.get_logger().info('Subscribe to /pepper/tablet_screen/display_image to change the display')
    
    def create_blank_image(self, width=800, height=1280):
        """Create a blank white image"""
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Add some text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Pepper Tablet"
        text_size = cv2.getTextSize(text, font, 2, 3)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        cv2.putText(image, text, (text_x, text_y), font, 2, (0, 0, 0), 3)
        
        return image
    
    def display_image_callback(self, msg):
        """Load and display an image from a file path"""
        image_path = msg.data
        self.get_logger().info(f'Loading image: {image_path}')
        
        try:
            # Load the image
            image = cv2.imread(image_path)
            
            if image is None:
                self.get_logger().error(f'Failed to load image: {image_path}')
                return
            
            # Resize to tablet dimensions (approximately 16:10 aspect ratio)
            image = cv2.resize(image, (800, 1280))
            
            self.current_image = image
            self.get_logger().info(f'Successfully loaded and displaying: {image_path}')
            
        except Exception as e:
            self.get_logger().error(f'Error loading image: {str(e)}')
    
    def publish_image(self):
        """Publish the current image"""
        try:
            # Convert OpenCV image to ROS Image message
            ros_image = self.bridge.cv2_to_imgmsg(self.current_image, encoding='bgr8')
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = 'tablet_screen'
            
            self.image_pub.publish(ros_image)
            
        except Exception as e:
            self.get_logger().error(f'Error publishing image: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = TabletScreenPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()