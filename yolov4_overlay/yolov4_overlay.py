#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
import numpy as np

class BlobDetectionNode(Node):
    def __init__(self):
        super().__init__('blob_detection_node')

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Subscribers
        self.create_subscription(
            Image,
            '/tb4_jazzy/oakd/rgb/preview/image_raw',
            self.image_callback,
            10
        )

        # Publisher for detections
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/blob_detections',
            10
        )

        # Setup SimpleBlobDetector parameters
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 500  # Minimum blob area (adjust for box size)
        params.maxArea = 10000  # Maximum blob area
        params.filterByCircularity = False  # Allow non-circular shapes for boxes
        params.filterByConvexity = True
        params.minConvexity = 0.8  # Ensure box-like shapes
        params.filterByInertia = True
        params.minInertiaRatio = 0.5  # Allow rectangular shapes
        self.detector = cv2.SimpleBlobDetector_create(params)

        self.get_logger().info("Blob Detection Node initialized")

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Convert to grayscale for blob detection
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Apply blob detection
        keypoints = self.detector.detect(gray_image)

        # Prepare Detection2DArray message
        detection_array = Detection2DArray()
        detection_array.header = msg.header

        # Process detected blobs
        for kp in keypoints:
            x, y = kp.pt
            size = kp.size  # Diameter of the blob
            w = h = size  # Approximate square bounding box

            # Filter for box-shaped blobs (aspect ratio heuristic)
            if self.is_box_shaped(w, h):
                detection = Detection2D()
                bbox = BoundingBox2D()

                # Bounding box coordinates
                bbox.center.position.x = float(x)
                bbox.center.position.y = float(y)
                bbox.size_x = float(w)
                bbox.size_y = float(h)

                # Object hypothesis
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.id = "box"
                hypothesis.score = 1.0  # Blob detector doesn't provide confidence
                hypothesis.pose.pose.position.x = float(x)
                hypothesis.pose.pose.position.y = float(y)

                detection.bbox = bbox
                detection.results.append(hypothesis)
                detection_array.detections.append(detection)

                # Draw blob as a rectangle
                x_min = int(x - w / 2)
                y_min = int(y - h / 2)
                x_max = int(x + w / 2)
                y_max = int(y + h / 2)
                cv2.rectangle(
                    cv_image,
                    (x_min, y_min),
                    (x_max, y_max),
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    cv_image,
                    "Box",
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

        # Publish detections
        self.detection_pub.publish(detection_array)

        # Display image (comment out if headless)
        cv2.imshow("Blob Detections", cv_image)
        cv2.waitKey(1)

    def is_box_shaped(self, w, h):
        # Heuristic for box-shaped blobs (near-square shapes)
        aspect_ratio = w / h if h > 0 else 1.0
        return 0.5 < aspect_ratio < 2.0

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = BlobDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
