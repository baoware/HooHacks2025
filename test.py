import cv2
import torch
import numpy as np
import math
from functions import get_roi_points, is_inside_roi, calculate_distance, check_incoming

# Load the local yolov5
model = torch.hub.load('./yolov5', 'yolov5s', source='local')
model.eval()

# Set of target obstacle classes (union from both scripts)
target_classes = {
    "person", "bench", "dog", "truck", "bus", "motorbike",
    "bicycle", "stop sign", "fire hydrant", "traffic light"
}

# Approximate object heights for distance estimation
known_heights = {
    "person": 1.7,   
    "bicycle": 1.0   
}

# Parameters for distance and approaching detection
alert_distance_person = 10.0  
focal_length = 1340            

# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

frame_delay = 100  # 100ms delay (~10 fps)
prev_persons = []  # For simple frame-to-frame association of person detections

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    height, width, _ = frame.shape

    # Get the ROI for the current frame
    roi_points = get_roi_points(width, height)

    # Run yolov5 detection
    results = model(frame)
    detections = results.xyxy[0]

    # For a very simple frame-to-frame association, store previous frame's people detections.
    current_persons = []

    for *box, conf, cls in detections:
        class_id = int(cls)
        label = model.names[class_id]

        # Process only targeted classes
        if label not in target_classes:
            continue

        x1, y1, x2, y2 = map(int, box)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Filter: process only detections inside the ROI
        if not is_inside_roi(center, roi_points):
            continue

        color = (0, 255, 0)  # default green for detections

        # For "person", compute distance and check if approaching
        if label == "person":
            bbox_height = y2 - y1
            distance = calculate_distance(label, bbox_height, focal_length, known_heights)
            if check_incoming(center, distance, prev_persons):
                # Change to red if approaching and within the alert distance threshold
                if distance <= alert_distance_person:
                    color = (0, 0, 255)
            current_persons.append((center[0], center[1], distance))
            text = f"{label} {conf:.2f} {distance:.1f}m"
        else:
            text = f"{label} {conf:.2f}"

        # Draw detection box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw trapezoid
    cv2.polylines(frame, [roi_points], isClosed=True, color=(0, 0, 255), thickness=2)

    # Update previous persons list
    prev_persons = current_persons

    cv2.imshow("Obstacle Detection", frame)
    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
