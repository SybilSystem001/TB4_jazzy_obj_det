#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from depthai_ros_msgs.msg import TrackDetection2DArray
from cv_bridge import CvBridge
import cv2

class YoloOverlayNode(Node):
    def __init__(self):
        super().__init__('yolov4_overlay_node')

        self.bridge = CvBridge()
        self.image = None
        self.detections = []

        # Subscribers
        self.create_subscription(Image, '/tb4_jazzy/oakd/rgb/preview/image_raw', self.image_callback, 10)
        self.create_subscription(TrackDetection2DArray, '/color/yolov4_tracklets', self.detections_callback, 10)

    def image_callback(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.draw_and_show()

    def detections_callback(self, msg):
        self.detections = msg.track_detections

    def draw_and_show(self):
        if self.image is None:
            return

        img = self.image.copy()

        for det in self.detections:
            x = det.roi.x
            y = det.roi.y
            w = det.roi.width
            h = det.roi.height
            label = det.label
            conf = det.confidence

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, f"{label} ({conf:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("YOLOv4 Detections", img)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = YoloOverlayNode()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
