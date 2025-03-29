import cv2
import torch
import numpy as np
import math

# Load the local yolov5
model = torch.hub.load('./yolov5', 'yolov5s', source='local')
model.eval()

# Set of target classes relevant for obstacles for blind pedestrians.
target_classes = {"person", "bench", "dog", "truck", "bus", "motorbike", "bicycle", "stop sign", "fire hydrant", "traffic light"}

# Webcam interface
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

