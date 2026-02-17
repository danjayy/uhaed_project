#!/usr/bin/env python3

"""
ASL Letter Recognition ROS 2 Humble Node

This node:
1. Reads camera feed from system camera
2. Detects hand landmarks using MediaPipe
3. Predicts ASL letter using multi-input ResNet18 model
4. Publishes predicted letter image to /output topic
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import String
from cv_bridge import CvBridge
import os
import json
from collections import Counter


class MultiInputASLModelResNet(nn.Module):
    """Multi-input model using ResNet18 for image features and landmarks."""
    
    def __init__(self, num_classes=26, pretrained=False):
        super(MultiInputASLModelResNet, self).__init__()
        
        # Landmark branch
        self.landmark_branch = nn.Sequential(
            nn.Linear(42, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Image branch - ResNet18
        resnet = models.resnet18(pretrained=pretrained)
        self.image_branch = nn.Sequential(*list(resnet.children())[:-1])
        self.image_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Fusion and classification layers
        self.classifier = nn.Sequential(
            nn.Linear(288, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, images, landmarks):
        landmark_features = self.landmark_branch(landmarks)
        image_features = self.image_branch(images)
        image_features = image_features.view(image_features.size(0), -1)
        image_features = self.image_fc(image_features)
        fused = torch.cat([landmark_features, image_features], dim=1)
        output = self.classifier(fused)
        return output

# ROS NODE FOR recognition
class ASLRecognitionNode(Node):
    def __init__(self):
        super().__init__('asl_recognition_node')
        
        self.get_logger().info("Initializing ASL Recognition Node...")
        
        # Declare parameters
        self.declare_parameter('confidence_threshold', 0.98)
        self.declare_parameter('history_size', 10)
        self.declare_parameter('camera_index', 0)
        
        # Get parameters
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.history_size = self.get_parameter('history_size').value
        camera_index = self.get_parameter('camera_index').value
        
        self.get_logger().info(f"Confidence threshold: {self.confidence_threshold}")
        self.get_logger().info(f"History size: {self.history_size}")
        
        # Get package directory (assumes files are in same directory as node)
        self.package_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Using device: {self.device}")
        
        # Initialize CvBridge
        self.bridge = CvBridge()
        
        # Initialize MediaPipe
        self.init_mediapipe()
        
        # Load class names
        self.load_class_names()
        
        # Load letter images
        self.load_letter_images()
        
        # Load model
        self.load_model()
        
        # Publishers
        self.image_pub = self.create_publisher(RosImage, '/output', 10)
        self.letter_pub = self.create_publisher(String, '/predicted_letter', 10)
        self.debug_pub = self.create_publisher(RosImage, '/debug_image', 10)
        
        self.get_logger().info("Publishers initialized:")
        self.get_logger().info("  - /output (Image): Predicted letter image")
        self.get_logger().info("  - /predicted_letter (String): Predicted letter text")
        self.get_logger().info("  - /debug_image (Image): Annotated camera feed")
        
        # Prediction smoothing
        self.prediction_history = []
        
        # Open camera
        self.get_logger().info(f"Opening camera {camera_index}...")
        self.camera = cv2.VideoCapture(camera_index)
        if not self.camera.isOpened():
            self.get_logger().error("Failed to open camera!")
            return
            
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Create timer for processing frames (30 Hz)
        self.timer = self.create_timer(1.0/30.0, self.timer_callback)
        
        self.get_logger().info("ASL Recognition Node initialized successfully!")
        
    def init_mediapipe(self):
        """Initialize MediaPipe hand detector."""
        model_path = os.path.join(self.package_dir, "mp_models/hand_landmarker.task")
        
        if not os.path.exists(model_path):
            self.get_logger().error(f"MediaPipe model not found at: {model_path}")
            raise FileNotFoundError(f"MediaPipe model not found: {model_path}")
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.get_logger().info("MediaPipe initialized")
        
    def load_class_names(self):
        """Load class names from JSON file."""
        class_names_path = os.path.join(self.package_dir, "class_names.json")
        
        try:
            with open(class_names_path, 'r') as f:
                class_data = json.load(f)
                self.class_names = class_data['class_names']
            self.get_logger().info(f"Loaded {len(self.class_names)} classes: {', '.join(self.class_names)}")
        except FileNotFoundError:
            self.get_logger().warn("class_names.json not found, using default A-Z")
            self.class_names = [chr(65+i) for i in range(26)]
            
    def load_letter_images(self):
        """Load letter images from directory."""
        letter_images_dir = os.path.join(self.package_dir, "letter_images")
        
        if not os.path.exists(letter_images_dir):
            self.get_logger().error(f"Letter images directory not found: {letter_images_dir}")
            raise FileNotFoundError(f"Letter images directory not found: {letter_images_dir}")
        
        self.letter_images = {}
        for class_name in self.class_names:
            img_path = os.path.join(letter_images_dir, f"{class_name}.jpeg")
            if os.path.exists(img_path):
                self.letter_images[class_name] = cv2.imread(img_path)
                self.get_logger().info(f"Loaded image for letter: {class_name}")
            else:
                self.get_logger().warn(f"Image not found for letter: {class_name}")
                # Create a placeholder image
                placeholder = np.zeros((400, 400, 3), dtype=np.uint8)
                cv2.putText(placeholder, class_name, (150, 250),
                           cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 10)
                self.letter_images[class_name] = placeholder
        
        self.get_logger().info(f"Loaded {len(self.letter_images)} letter images")
        
    def load_model(self):
        """Load the trained PyTorch model."""
        model_path = os.path.join(self.package_dir, "asl_rgb_model.pth")
        
        if not os.path.exists(model_path):
            self.get_logger().error(f"Model not found at: {model_path}")
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = MultiInputASLModelResNet(num_classes=len(self.class_names)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.get_logger().info("Model loaded successfully")
        
    def normalize_landmarks(self, landmarks):
        """Normalize landmarks relative to wrist position and hand size."""
        landmarks = landmarks.reshape(21, 2)
        wrist = landmarks[0].copy()
        landmarks = landmarks - wrist
        hand_size = np.linalg.norm(landmarks[12])
        if hand_size > 0:
            landmarks = landmarks / hand_size
        return landmarks.flatten()
        
    def extract_landmarks(self, image):
        """Extract hand landmarks from image."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = self.detector.detect(mp_image)
        if results.hand_landmarks:
            landmarks = results.hand_landmarks[0]
            coords = [[lm.x, lm.y] for lm in landmarks]
            raw_landmarks = np.array(coords).flatten()
            normalized_landmarks = self.normalize_landmarks(raw_landmarks)
            return normalized_landmarks, results
        else:
            return None, None
            
    def extract_hand_roi(self, frame, hand_landmarks, padding=60):
        """Extract a bounding box around the detected hand."""
        h, w, _ = frame.shape
        x_coords = [lm.x * w for lm in hand_landmarks]
        y_coords = [lm.y * h for lm in hand_landmarks]
        
        x_min = max(0, int(min(x_coords)) - padding)
        x_max = min(w, int(max(x_coords)) + padding)
        y_min = max(0, int(min(y_coords)) - padding)
        y_max = min(h, int(max(y_coords)) + padding)
        
        roi = frame[y_min:y_max, x_min:x_max]
        
        # Make square
        roi_h, roi_w = roi.shape[:2]
        if roi_h > roi_w:
            diff = roi_h - roi_w
            pad_left = diff // 2
            pad_right = diff - pad_left
            roi = cv2.copyMakeBorder(roi, 0, 0, pad_left, pad_right, 
                                    cv2.BORDER_CONSTANT, value=[0, 0, 0])
        elif roi_w > roi_h:
            diff = roi_w - roi_h
            pad_top = diff // 2
            pad_bottom = diff - pad_top
            roi = cv2.copyMakeBorder(roi, pad_top, pad_bottom, 0, 0, 
                                    cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        return roi, (x_min, y_min, x_max, y_max)
        
    def preprocess_roi(self, roi):
        """Preprocess ROI to match training data."""
        roi = cv2.GaussianBlur(roi, (3, 3), 0)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        roi = cv2.merge([l, a, b])
        roi = cv2.cvtColor(roi, cv2.COLOR_LAB2BGR)
        return roi
        
    def preprocess_image_for_model(self, image):
        """Convert image to tensor for model input."""
        image = self.preprocess_roi(image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # applying transform to be same as that used for validation during training
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transform(pil_image).unsqueeze(0)
        
    def timer_callback(self):
        """Process camera frames at regular intervals."""
        ret, frame = self.camera.read()
        if not ret:
            self.get_logger().warn("Failed to read frame from camera")
            return
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Extract landmarks
        landmark_features, results = self.extract_landmarks(frame)
        
        if landmark_features is not None and results.hand_landmarks:
            # Extract hand ROI
            hand_roi, bbox = self.extract_hand_roi(frame, results.hand_landmarks[0])
            
            # Preprocess inputs
            image_input = self.preprocess_image_for_model(hand_roi).to(self.device)
            landmark_input = torch.FloatTensor(landmark_features).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_input, landmark_input)
                probabilities = torch.softmax(outputs, dim=1)
                top3_probs, top3_indices = torch.topk(probabilities[0], 3)
                
                predicted_class = top3_indices[0].item()
                confidence = top3_probs[0].item()
                predicted_letter = self.class_names[predicted_class]
                
                # Apply confidence threshold
                if confidence >= self.confidence_threshold:
                    self.prediction_history.append(predicted_class)
                    if len(self.prediction_history) > self.history_size:
                        self.prediction_history.pop(0)
                    
                    # Smooth predictions
                    if len(self.prediction_history) >= 3:
                        smoothed_prediction = Counter(self.prediction_history).most_common(1)[0][0]
                        final_letter = self.class_names[smoothed_prediction]
                    else:
                        final_letter = predicted_letter
                    
                    # Publish letter image
                    if final_letter in self.letter_images:
                        letter_img = self.letter_images[final_letter]
                        try:
                            ros_image = self.bridge.cv2_to_imgmsg(letter_img, encoding="bgr8")
                            self.image_pub.publish(ros_image)
                        except Exception as e:
                            self.get_logger().error(f"Error publishing image: {str(e)}")
                    
                    # Publish letter text
                    letter_msg = String()
                    letter_msg.data = final_letter
                    self.letter_pub.publish(letter_msg)
                    
                    self.get_logger().info(f"Published: {final_letter} ({confidence:.2%})")
                else:
                    self.get_logger().debug(f"Low confidence: {confidence:.2%}")
            
            # Publish debug image with annotations
            debug_frame = frame.copy()
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(debug_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            if confidence >= self.confidence_threshold:
                color = (0, 255, 0)
                status = f"{final_letter}: {confidence:.1%}"
            else:
                color = (0, 165, 255)
                status = f"?: {confidence:.1%}"
            
            cv2.putText(debug_frame, status, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            try:
                debug_ros_image = self.bridge.cv2_to_imgmsg(debug_frame, encoding="bgr8")
                self.debug_pub.publish(debug_ros_image)
            except Exception as e:
                self.get_logger().error(f"Error publishing debug image: {str(e)}")
        else:
            self.prediction_history.clear()
            
    def destroy_node(self):
        """Clean up resources."""
        if hasattr(self, 'camera'):
            self.camera.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = ASLRecognitionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()