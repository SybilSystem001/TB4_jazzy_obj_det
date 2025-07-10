#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
import numpy as np
import depthai as dai

class YoloBoxDetectionNode(Node):
    def __init__(self):
        super().__init__('yolo_box_detection_node')

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Load YOLOv4-tiny model
        self.net = cv2.dnn.readNetFromDarknet(
            '/root/ros2_ws/src/yolov4_overlay/yolov4/yolov4-tiny.cfg',
            '/root/ros2_ws/src/yolov4_overlay/yolov4/yolov4-tiny.weights'
        )
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Use CPU for Raspberry Pi
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        # Load COCO class names
        with open('/root/ros2_ws/src/yolov4_overlay/yolov4/coco.names', 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

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

        self.get_logger().info("YOLOv4 Box Detection Node initialized")

    def setup_oakd_pipeline(self):
        cam_rgb = self.pipeline.create(dai.node.ColorCamera)
        xout_rgb = self.pipeline.create(dai.node.XLinkOut)

        cam_rgb.setPreviewSize(416, 416)  # YOLOv4-tiny input size
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Prepare image for YOLOv4
        blob = cv2.dnn.blobFromImage(cv_image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        # Process detections
        detection_array = Detection2DArray()
        detection_array.header = msg.header

        height, width = cv_image.shape[:2]
        boxes = []
        confidences = []
        class_ids = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Confidence threshold
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    if self.is_box_shaped(x, y, w, h):
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

        # Non-Max Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in indices:
            box = boxes[i]
            x, y, w, h = box
            detection = Detection2D()
            bbox = BoundingBox2D()
            
            bbox.center.position.x = float(x + w / 2)
            bbox.center.position.y = float(y + h / 2)
            bbox.size_x = float(w)
            bbox.size_y = float(h)

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = str(class_ids[i])
            hypothesis.score = confidences[i]
            hypothesis.pose.pose.position.x = bbox.center.position.x
            hypothesis.pose.pose.position.y = bbox.center.position.y

            detection.bbox = bbox
            detection.results.append(hypothesis)
            detection_array.detections.append(detection)

            # Draw bounding box
            label = f"{self.classes[class_ids[i]]} ({confidences[i]:.2f})"
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                cv_image,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

        # Publish detections
        self.detection_pub.publish(detection_array)

        # Display (comment out if headless)
        cv2.imshow("Box Detections", cv_image)
        cv2.waitKey(1)

    def is_box_shaped(self, x, y, w, h):
        aspect_ratio = w / h if h > 0 else 1.0
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
