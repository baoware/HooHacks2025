import cv2
import torch
import numpy as np

# Load the local yolov5
model = torch.hub.load('./yolov5', 'yolov5s', source='local')
model.eval()  

# Set of target classes relevant for obstacles for blind pedestrians.
target_classes = {"person", "bench", "dog", "truck", "bus", "motorbike", "bicycle"}

# Webcam interface
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

# Delay between frames (200ms ~ 5 fps)
frame_delay = 200

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Run yolov5 detection
    results = model(frame)
    
    detections = results.xyxy[0]
    for *box, conf, cls in detections:
        class_id = int(cls)
        label = model.names[class_id]
        # Only process detections for our target obstacle classes.
        if label in target_classes:
            x1, y1, x2, y2 = map(int, box)
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    cv2.imshow("Obstacle Detection", frame)

    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()