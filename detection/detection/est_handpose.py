import cv2
import numpy as np

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                               # Palm
]


def draw_landmarks_on_image(rgb_image, detection_result):
    annotated_image = rgb_image.copy()
    h, w, _ = annotated_image.shape

    for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
        # Draw landmarks
        for lm in hand_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated_image, (cx, cy), 5, (0, 255, 0), -1)

        # Draw connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            start = hand_landmarks[start_idx]
            end = hand_landmarks[end_idx]
            x1, y1 = int(start.x * w), int(start.y * h)
            x2, y2 = int(end.x * w), int(end.y * h)
            cv2.line(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Handedness label
        handedness = detection_result.handedness[hand_idx][0].category_name
        x_coords = [lm.x for lm in hand_landmarks]
        y_coords = [lm.y for lm in hand_landmarks]

        text_x = int(min(x_coords) * w)
        text_y = int(min(y_coords) * h) - MARGIN

        cv2.putText(
            annotated_image,
            handedness,
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA
        )

    return annotated_image
