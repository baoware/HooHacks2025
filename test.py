import cv2
import torch
import numpy as np
import math
import time
from functions import get_roi_points, is_inside_roi, calculate_distance, check_incoming
from alerts import alert_approaching, alert_incoming

# Load the local yolov5
model = torch.hub.load('./yolov5', 'yolov5n', source='local')
model.eval()

# Set of target obstacle classes (union from both scripts)
target_incoming_classes = {
    "person", "dog", "bicycle"
}

target_approaching_classed = {
    "stop sign", "traffic light"
}

# Approximate object heights for distance estimation
known_heights = {
    "person": 1.7,   
    "bicycle": 1.0   
}

# Parameters for distance and approaching detection
alert_distance_person = 10.0  
focal_length = 1340            

last_person_alert_time = 0
alert_interval = 15

# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

frame_delay = 30  # 100ms delay (~10 fps)
prev_persons = []  # For simple frame-to-frame association of person detections

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    current_time = time.time()
    
    height, width, _ = frame.shape

    # Get the ROI for the current frame
    roi_points = get_roi_points(width, height)

    # Run yolov5 detection
    results = model(frame)
    detections = results.xyxy[0]

    # For a very simple frame-to-frame association, store previous frame's people detections.
    current_persons = []
    alerted_persons = []

    
    for *box, conf, cls in detections:
        if conf < 0.6:
            continue

        class_id = int(cls)
        label = model.names[class_id]

        # Process only targeted classes
        if label not in target_incoming_classes:
            continue

        x1, y1, x2, y2 = map(int, box)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        if not is_inside_roi(center, roi_points):
            continue

        color = (0, 255, 0)  # default green

        if label == "person":
            bbox_height = y2 - y1
            distance = calculate_distance(label, bbox_height, focal_length, known_heights)
            
            # Check if this person is incoming and has not been alerted yet.
            if distance <= alert_distance_person and (current_time - last_person_alert_time) >= alert_interval:
                if distance <= alert_distance_person:
                    color = (0, 0, 255)  # alert: change to red
                    alerted_persons.append((center[0], center[1], distance))
                    last_person_alert_time = current_time
                    alert_incoming(label)
                
            
            current_persons.append((center[0], center[1], distance))
            text = f"{label} {conf:.2f} {distance:.1f}m"
        else:
            text = f"{label} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw trapezoid
    cv2.polylines(frame, [roi_points], isClosed=True, color=(0, 0, 255), thickness=2)

    # Update previous persons list
    prev_persons = current_persons

    cv2.imshow("Obstacle Detection", frame)
    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
