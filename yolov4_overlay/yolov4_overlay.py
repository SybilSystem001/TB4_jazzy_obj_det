#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import depthai as dai

class YoloBoxDetectionNode(Node):
    def __init__(self):
        super().__init__('yolo_box_detection_node')

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Load YOLOv8 Nano model (pre-trained on COCO dataset)
        self.model = YOLO('yolov8n.pt')  # Download from Ultralytics if not already present

        # Subscribers and Publishers
        self.image_sub = self.create_subscription(
            Image,
            '/tb4_jazzy/oakd/rgb/preview/image_raw',
            self.image_callback,
            10
        )
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/box_detections',
            10
        )

        # OAK-D-Pro pipeline setup
        self.pipeline = dai.Pipeline()
        self.setup_oakd_pipeline()

        # Initialize OAK-D device
        self.device = dai.Device(self.pipeline)
        self.rgb_queue = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        self.get_logger().info("YOLO Box Detection Node initialized")

    def setup_oakd_pipeline(self):
        # Create nodes for OAK-D-Pro
        cam_rgb = self.pipeline.create(dai.node.ColorCamera)
        xout_rgb = self.pipeline.create(dai.node.XLinkOut)

        # Set camera properties
        cam_rgb.setPreviewSize(416, 416)  # YOLOv8 input size
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        # Set output stream
        xout_rgb.setStreamName("rgb")

        # Link nodes
        cam_rgb.preview.link(xout_rgb.input)

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Run YOLOv8 inference
        results = self.model(cv_image, conf=0.5)  # Confidence threshold of 0.5

        # Prepare Detection2DArray message
        detection_array = Detection2DArray()
        detection_array.header = msg.header

        # Process detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                label = self.model.names[cls]
                
                # Filter for box-shaped objects (e.g., COCO classes like 'cardboard box' or similar)
                # COCO dataset doesn't have a direct "box" class, so we'll use a heuristic for rectangular objects
                # For simplicity, we'll assume objects with rectangular bounding boxes
                if self.is_box_shaped(box, cv_image):
                    detection = Detection2D()
                    bbox = BoundingBox2D()
                    
                    # Bounding box coordinates
                    x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
                    bbox.center.position.x = float((x_min + x_max) / 2)
                    bbox.center.position.y = float((y_min + y_max) / 2)
                    bbox.size_x = float(x_max - x_min)
                    bbox.size_y = float(y_max - y_min)

                    # Object hypothesis
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.id = str(cls)
                    hypothesis.score = float(box.conf[0])
                    hypothesis.pose.pose.position.x = bbox.center.position.x
                    hypothesis.pose.pose.position.y = bbox.center.position.y

                    detection.bbox = bbox
                    detection.results.append(hypothesis)
                    detection_array.detections.append(detection)

                    # Draw bounding box for visualization
                    cv2.rectangle(
                        cv_image,
                        (int(x_min), int(y_min)),
                        (int(x_max), int(y_max)),
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        cv_image,
                        f"{label} ({hypothesis.score:.2f})",
                        (int(x_min), int(y_min - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2
                    )

        # Publish detections
        self.detection_pub.publish(detection_array)

        # Display the image with detections
        cv2.imshow("Box Detections", cv_image)
        cv2.waitKey(1)

    def is_box_shaped(self, box, image):
        # Heuristic to identify box-shaped objects based on bounding box aspect ratio
        x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
        width = x_max - x_min
        height = y_max - y_min
        aspect_ratio = width / height if height > 0 else 1.0

        # Consider objects with aspect ratio close to 1 (square-like) or within a reasonable range
        # This is a simple heuristic; adjust based on your specific needs
        return 0.5 < aspect_ratio < 2.0

    def destroy_node(self):
        self.device.close()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = YoloBoxDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
