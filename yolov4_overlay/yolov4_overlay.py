#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import argparse

def setup_blob_detector():
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 200
    params.maxArea = 20000
    params.filterByCircularity = False
    params.filterByConvexity = True
    params.minConvexity = 0.7
    params.filterByInertia = True
    params.minInertiaRatio = 0.3
    return cv2.SimpleBlobDetector_create(params)

class BlobDetectionNode(Node):
    def __init__(self):
        super().__init__('blob_detection_node')
        self.bridge = CvBridge()
        self.detector = self.setup_blob_detector()

        self.subscription = self.create_subscription(
            Image,
            '/tb4_jazzy/oakd/rgb/preview/image_raw',
            self.image_callback,
            10
        )
        self.get_logger().info('Running in ROS 2 mode (subscribed to camera topic)')


    def process_frame(self, cv_image):
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        keypoints = self.detector.detect(gray)

        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            size = kp.size
            cv2.circle(cv_image, (x, y), int(size/2), (0, 255, 0), 2)

        cv2.imshow("Blob Detection", cv_image)
        cv2.waitKey(1)

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.process_frame(cv_image)

def local_camera_mode():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open local camera.")
        return

    detector = setup_blob_detector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        keypoints = detector.detect(gray)

        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            size = kp.size
            cv2.circle(frame, (x, y), int(size/2), (0, 255, 0), 2)

        cv2.imshow('Blob Detection - Local Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Blob Detection Node with ROS 2 and Local Camera Support')
    parser.add_argument('--mode', type=str, choices=['ros', 'local'], default='ros',
                        help='Mode to run: ros (subscribe to ROS topic) or local (use USB camera)')
    args = parser.parse_args()

    if args.mode == 'ros':
        rclpy.init()
        node = BlobDetectionNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
            cv2.destroyAllWindows()
    else:
        local_camera_mode()

if __name__ == '__main__':
    main()

