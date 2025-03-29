import cv2
import torch
import numpy as np
import math

# Load the local yolov5
model = torch.hub.load('./yolov5', 'yolov5s', source='local')
model.eval()

# Set of target classes relevant for obstacles for blind pedestrians.
target_classes = {"person", "bicycle"}

# Approximate object heights
known_heights = {
    "person": 1.7,   
    "bicycle": 1.0   
}

# Early alert threshold for person in meters (10 meters)
alert_distance_person = 10.0

# Assumed focal length in pixels (adjust via calibration)
focal_length = 700

# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

frame_delay = 200  # 200ms delay (~5 fps)

# For a very simple frame-to-frame association, store previous frame's people detections.
prev_persons = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    #  # Run yolov5 detection
    results = model(frame)
    detections = results.xyxy[0]
    current_persons = [] 

    # Process each detection
    for *box, conf, cls in detections:
        class_id = int(cls)
        label = model.names[class_id]
        x1, y1, x2, y2 = map(int, box)
        bbox_height = y2 - y1

        if bbox_height <= 0:
            continue

        # Estimate distance using a pinhole camera model
        estimated_distance = (known_heights.get(label, 1.0) * focal_length) / bbox_height

        # Compute box center
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        color = (0, 255, 0)

        # If the object is a person, check if they are moving towards you
        if label == "person":
            # Attempt to find a matching detection in the previous frame
            approaching = False
            min_dist = float("inf")
            matching_prev = None
            for (prev_cx, prev_cy, prev_dist) in prev_persons:
                diff = math.hypot(center_x - prev_cx, center_y - prev_cy)
                if diff < min_dist and diff < 50:  
                    min_dist = diff
                    matching_prev = prev_dist

            # If a matching previous detection was found, compare distances.
            # If the current estimated distance is smaller (object got closer), mark as approaching.
            if matching_prev is not None and estimated_distance < matching_prev - 0.1:
                approaching = True

            # If approaching and within the alert threshold (10 meters), change the color to red.
            if approaching and estimated_distance <= alert_distance_person:
                color = (0, 0, 255)

            # Store this person detection
            current_persons.append((center_x, center_y, estimated_distance))

        # Draw boxes
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f} {estimated_distance:.1f}m"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Update the previous persons list
    prev_persons = current_persons

    cv2.imshow("Distance and Motion Detection", frame)
    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
