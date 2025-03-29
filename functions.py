import numpy as np
import math
import cv2

def get_roi_points(width, height):
    bottom_left = (int(0.10 * width), height)
    bottom_right = (int(0.90 * width), height)
    top_left = (int(0.40 * width), int(0.10 * height))
    top_right = (int(0.60 * width), int(0.10 * height))
    roi_points = np.array([bottom_left, bottom_right, top_right, top_left], np.int32)
    return roi_points.reshape((-1, 1, 2))

def is_inside_roi(point, roi_points):
    return cv2.pointPolygonTest(roi_points, point, False) >= 0

def calculate_distance(label, bbox_height, focal_length, known_heights):
    object_height = known_heights.get(label, 1.0)
    if bbox_height <= 0:
        return float('inf')
    return (object_height * focal_length) / bbox_height

def check_incoming(center, current_distance, previous_detections, movement_threshold=50, distance_delta=0.01):
    center_x, center_y = center
    min_diff = float("inf")
    matching_distance = None
    for (prev_x, prev_y, prev_distance) in previous_detections:
        diff = math.hypot(center_x - prev_x, center_y - prev_y)
        if diff < min_diff and diff < movement_threshold:
            min_diff = diff
            matching_distance = prev_distance
    if matching_distance is not None and current_distance < matching_distance - distance_delta:
        return True
    return False










