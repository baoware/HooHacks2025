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

    height, width, _ = frame.shape
    
    # Define trapezoidal region of interest
    bottom_left = (int(0.25*width), height)
    bottom_right = (int(0.75*width), height)

    # Adjust as needed: top edge is at ~33% of the height, with horizontal margins of 40% of width on each side.
    top_left = (int(0.40 * width), int(0.33 * height))
    top_right = (int(0.60 * width), int(0.33 * height))
    
    roi_points = np.array([bottom_left, bottom_right, top_right, top_left], np.int32)
    roi_points = roi_points.reshape((-1, 1, 2))

    # Run yolov5 detection
    results = model(frame)
    detections = results.xyxy[0]
    for *box, conf, cls in detections:
        class_id = int(cls)
        label = model.names[class_id]
        # Only process detections for our target obstacle classes.
        if label in target_classes:
            x1, y1, x2, y2 = map(int, box)  

            # Draw detection box and label
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            if cv2.pointPolygonTest(roi_points, (center_x, center_y), False) >= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    # Draw trapezoid
    cv2.polylines(frame, [roi_points], isClosed=True, color=(0, 0, 255), thickness=2)

    cv2.imshow("Obstacle Detection", frame)

    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
