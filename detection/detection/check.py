import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
from est_handpose import *  # helper

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize MediaPipe
MODEL_PATH = "mp_models/hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(options)


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
        # Remove the final fully connected layer
        self.image_branch = nn.Sequential(*list(resnet.children())[:-1])
        # ResNet18 outputs 512 features
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
        # Process landmark branch
        landmark_features = self.landmark_branch(landmarks)
        
        # Process image branch
        image_features = self.image_branch(images)
        image_features = image_features.view(image_features.size(0), -1)
        image_features = self.image_fc(image_features)
        
        # Concatenate features
        fused = torch.cat([landmark_features, image_features], dim=1)
        
        # Classification
        output = self.classifier(fused)
        return output


# ============================================================================
# SOLUTION 1: Normalize landmarks relative to hand size and position
# ============================================================================
def normalize_landmarks(landmarks):
    """
    Normalize landmarks relative to wrist position and hand size.
    This makes landmarks invariant to hand position and scale.
    """
    landmarks = landmarks.reshape(21, 2)
    
    # Center at wrist (landmark 0)
    wrist = landmarks[0].copy()
    landmarks = landmarks - wrist
    
    # Scale by hand size (distance from wrist to middle finger tip)
    hand_size = np.linalg.norm(landmarks[12])  # Middle finger tip (landmark 12)
    if hand_size > 0:
        landmarks = landmarks / hand_size
    
    return landmarks.flatten()


def extract_landmarks(image):
    """Extract hand landmarks from image."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image_rgb = np.ascontiguousarray(image_rgb.astype(np.uint8))
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )

    results = detector.detect(mp_image)
    
    if results.hand_landmarks:
        landmarks = results.hand_landmarks[0]
        coords = [[lm.x, lm.y] for lm in landmarks]
        raw_landmarks = np.array(coords).flatten()
        
        # ============================================================================
        # SOLUTION 1 APPLIED: Normalize landmarks to match training data
        # ============================================================================
        normalized_landmarks = normalize_landmarks(raw_landmarks)
        
        return normalized_landmarks, results
    else:
        return None, None


# ============================================================================
# SOLUTION 2: Preprocess ROI to match training data appearance
# ============================================================================
def preprocess_roi_for_consistency(roi):
    """
    Preprocess camera ROI to be more similar to training data.
    This reduces domain gap between clean training images and noisy camera feed.
    """
    # Apply slight Gaussian blur to reduce noise (training images are cleaner)
    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    
    # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This helps normalize lighting differences
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    roi = cv2.merge([l, a, b])
    roi = cv2.cvtColor(roi, cv2.COLOR_LAB2BGR)
    
    return roi


def preprocess_image_for_model(image):
    """
    Convert captured image to tensor for ResNet input.
    CRITICAL: Must match validation transform exactly!
    """
    # ============================================================================
    # SOLUTION 2 APPLIED: Preprocess to match training data
    # ============================================================================
    image = preprocess_roi_for_consistency(image)
    
    # Convert BGR to RGB (OpenCV uses BGR, PIL/PyTorch expects RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image_rgb)
    
    # Define transform - MUST match validation transform exactly
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Converts to [0, 1] range and changes to CxHxW format
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transform and add batch dimension
    image_tensor = transform(pil_image).unsqueeze(0)
    
    return image_tensor


# ============================================================================
# SOLUTION 3: Extract hand ROI for better focus
# ============================================================================
def extract_hand_roi(frame, hand_landmarks, padding=50):
    """
    Extract a bounding box around the detected hand.
    This focuses on the hand region similar to training data.
    """
    h, w, _ = frame.shape
    
    # Get all landmark coordinates
    x_coords = [lm.x * w for lm in hand_landmarks]
    y_coords = [lm.y * h for lm in hand_landmarks]
    
    # Find bounding box
    x_min = max(0, int(min(x_coords)) - padding)
    x_max = min(w, int(max(x_coords)) + padding)
    y_min = max(0, int(min(y_coords)) - padding)
    y_max = min(h, int(max(y_coords)) + padding)
    
    # Extract ROI
    roi = frame[y_min:y_max, x_min:x_max]
    
    # Make square (training images are square)
    roi_h, roi_w = roi.shape[:2]
    if roi_h > roi_w:
        diff = roi_h - roi_w
        pad_left = diff // 2
        pad_right = diff - pad_left
        roi = cv2.copyMakeBorder(roi, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    elif roi_w > roi_h:
        diff = roi_w - roi_h
        pad_top = diff // 2
        pad_bottom = diff - pad_top
        roi = cv2.copyMakeBorder(roi, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    return roi, (x_min, y_min, x_max, y_max)


def main():
    # Load class names from training
    import json
    try:
        with open('class_names.json', 'r') as f:
            class_data = json.load(f)
            CLASS_NAMES = class_data['class_names']
        print(f"Loaded {len(CLASS_NAMES)} classes from training: {', '.join(CLASS_NAMES)}")
    except FileNotFoundError:
        print("Warning: class_names.json not found, using default A-Z")
        CLASS_NAMES = [chr(65+i) for i in range(26)]  # A-Z
    
    # Load model
    print("\nLoading model...")
    model = MultiInputASLModelResNet(num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load("best_asl_rgb_model.pth", map_location=device))
    
    # ============================================================================
    # CRITICAL: Set model to evaluation mode
    # ============================================================================
    model.eval()  # Disables dropout and sets batch norm to eval mode
    
    print(f"Model loaded successfully on {device}!")
    print(f"Recognizing {len(CLASS_NAMES)} classes")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set camera resolution for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\n=== ASL Letter Recognition (RGB + ResNet18) ===")
    print("Press 'q' to quit")
    print("Press 's' to save a test frame for debugging")
    print("Show your hand sign to the camera\n")
    
    # ============================================================================
    # SOLUTION 4: Add smoothing for more stable predictions
    # ============================================================================
    prediction_history = []
    history_size = 10  # Average over last 5 frames
    
    with torch.no_grad():  # Disable gradient computation for inference
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Extract landmarks from full frame first
            landmark_features, results = extract_landmarks(frame)
            
            if landmark_features is not None and results.hand_landmarks:
                # ============================================================================
                # SOLUTION 3 APPLIED: Extract hand ROI for focused processing
                # ============================================================================
                hand_roi, bbox = extract_hand_roi(frame, results.hand_landmarks[0], padding=60)
                x_min, y_min, x_max, y_max = bbox
                
                # Draw hand landmarks on original frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame = draw_landmarks_on_image(rgb_frame, results)
                frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                
                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # ============================================================================
                # SOLUTION 2 & 3 APPLIED: Preprocess hand ROI
                # ============================================================================
                # Preprocess the hand ROI (not the full frame)
                image_input = preprocess_image_for_model(hand_roi).to(device)
                landmark_input = torch.FloatTensor(landmark_features).unsqueeze(0).to(device)
                
                # Make prediction
                outputs = model(image_input, landmark_input)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get top 3 predictions
                top3_probs, top3_indices = torch.topk(probabilities[0], 3)
                
                predicted_class = top3_indices[0].item()
                confidence = top3_probs[0].item()
                predicted_letter = CLASS_NAMES[predicted_class]
                
                # ============================================================================
                # SOLUTION 5: Confidence Threshold - Only show prediction if confident
                # ============================================================================
                CONFIDENCE_THRESHOLD = 0.98  # 98% confidence required
                
                if confidence < CONFIDENCE_THRESHOLD:
                    # Not confident enough - show "uncertain"
                    display_letter = "?"
                    display_confidence = confidence
                    is_confident = False
                else:
                    # Confident prediction
                    display_letter = predicted_letter
                    display_confidence = confidence
                    is_confident = True
                
                # ============================================================================
                # SOLUTION 4 APPLIED: Smooth predictions over time
                # ============================================================================
                # Only add to history if confident
                if is_confident:
                    prediction_history.append(predicted_class)
                    if len(prediction_history) > history_size:
                        prediction_history.pop(0)
                
                # Use mode (most common) prediction for stability
                if len(prediction_history) >= 3:
                    from collections import Counter
                    smoothed_prediction = Counter(prediction_history).most_common(1)[0][0]
                    smoothed_letter = CLASS_NAMES[smoothed_prediction]
                else:
                    smoothed_letter = display_letter
                
                # Display main prediction (smoothed)
                text = f"{smoothed_letter}"
                conf_text = f"{display_confidence:.1%}"
                
                # Draw prediction box with color based on confidence
                box_color = (0, 255, 0) if is_confident else (0, 165, 255)  # Green if confident, Orange if not
                cv2.rectangle(frame, (10, 10), (250, 120), (0, 0, 0), -1)
                cv2.rectangle(frame, (10, 10), (250, 120), box_color, 3)
                
                cv2.putText(frame, text, (30, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, box_color, 4)
                cv2.putText(frame, conf_text, (30, 105),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show threshold warning if not confident
                if not is_confident:
                    cv2.putText(frame, "Low confidence", (10, 135),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                
                # Display top 3 predictions
                y_offset = 150
                cv2.putText(frame, "Top 3:", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                for i in range(3):
                    pred_idx = top3_indices[i].item()
                    pred_prob = top3_probs[i].item()
                    pred_letter = CLASS_NAMES[pred_idx]
                    
                    text = f"{i+1}. {pred_letter}: {pred_prob:.1%}"
                    cv2.putText(frame, text, (10, y_offset + 30 + i*25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show if smoothing is active
                if smoothed_letter != predicted_letter:
                    cv2.putText(frame, f"(Raw: {predicted_letter})", (10, y_offset + 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                # Print to terminal
                terminal_text = f"Prediction: {smoothed_letter} ({confidence:.2%})"
                terminal_text += f" | Top 3: "
                for i in range(3):
                    pred_letter = CLASS_NAMES[top3_indices[i].item()]
                    pred_prob = top3_probs[i].item()
                    terminal_text += f"{pred_letter}({pred_prob:.1%}) "
                
                print(f"\r{terminal_text}", end='', flush=True)
            else:
                prediction_history.clear()  # Reset history when no hand detected
                
                cv2.rectangle(frame, (10, 10), (350, 80), (0, 0, 0), -1)
                cv2.rectangle(frame, (10, 10), (350, 80), (0, 0, 255), 3)
                cv2.putText(frame, "No hand detected", (20, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Add instructions
            cv2.putText(frame, "Press 'q' to quit | 's' to save", (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('ASL Letter Recognition (RGB + ResNet18)', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and landmark_features is not None:
                # ============================================================================
                # DEBUG FEATURE: Save test frame and preprocessed input
                # ============================================================================
                cv2.imwrite('debug_frame.jpg', frame)
                cv2.imwrite('debug_roi.jpg', hand_roi)
                print("\n[DEBUG] Saved debug_frame.jpg and debug_roi.jpg")
                
                # Save what the model actually sees
                import torchvision
                torchvision.utils.save_image(
                    image_input, 
                    'debug_model_input.jpg', 
                    normalize=True,
                    value_range=(-2.5, 2.5)  # Approximate range after normalization
                )
                print("[DEBUG] Saved debug_model_input.jpg (what model sees)")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n\nExiting...")


if __name__ == "__main__":
    main()