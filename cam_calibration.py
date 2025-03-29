import cv2
import numpy as np
import math

def basic_camera_test(cap, test_duration=3000):
    """
    Displays the camera feed for a short duration (in milliseconds)
    so you can verify that the camera is working.
    """
    import time
    start_time = time.time()
    while int((time.time() - start_time) * 1000) < test_duration:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame during camera test.")
            break
        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow("Camera Test")
    print("Camera test complete. If the window was black, check your camera and settings.")

def click_event(event, x, y, flags, params):
    """
    Mouse callback function that records the points where the user clicks.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        params['points'].append((x, y))
        cv2.circle(params['image'], (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Calibration", params['image'])

def calibrate_with_ruler(cap, known_length_m=0.4572):
    """
    Calibrates the camera's focal length using an 18-inch ruler.
    
    Parameters:
        cap: cv2.VideoCapture object.
        known_length_m: The physical length of the ruler in meters (default is 18 inches ~ 0.4572 m).
    
    Returns:
        focal_length: Estimated focal length in pixels, or None if calibration fails.
    """
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame for calibration.")
        return None
    
    # Display the captured frame for calibration
    clone = frame.copy()
    cv2.imshow("Calibration", clone)
    print("Place the 18-inch ruler in view.")
    print("Click on the two endpoints of the ruler in the image window.")
    
    # Prepare a parameters dictionary to store clicked points
    params = {'points': [], 'image': clone}
    cv2.setMouseCallback("Calibration", click_event, params)
    
    # Wait until two points have been selected or user quits
    while True:
        cv2.imshow("Calibration", params['image'])
        key = cv2.waitKey(1) & 0xFF
        if len(params['points']) >= 2:
            break
        if key == ord('q'):
            break

    cv2.destroyWindow("Calibration")
    
    if len(params['points']) < 2:
        print("Calibration aborted: insufficient points selected.")
        return None
    
    # Calculate the pixel distance between the two clicked points
    pt1, pt2 = params['points'][:2]
    pixel_distance = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
    print(f"Measured pixel distance: {pixel_distance:.2f} pixels")
    
    # Prompt user for the distance from the camera to the ruler (in meters)
    try:
        known_distance = float(input("Enter the distance from the camera to the ruler (in meters): "))
    except ValueError:
        print("Invalid input. Calibration aborted.")
        return None
    
    # Calculate the focal length using the pinhole camera formula
    focal_length = (pixel_distance * known_distance) / known_length_m
    print(f"Estimated focal length: {focal_length:.2f} pixels")
    
    return focal_length

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
    else:
        # Run a basic camera test to ensure the feed is working
        basic_camera_test(cap, test_duration=3000)
        
        # Proceed with calibration after the camera test
        fl = calibrate_with_ruler(cap)
        if fl is not None:
            print(f"Calibration complete. Focal length: {fl:.2f} pixels")
        cap.release()
    cv2.destroyAllWindows()

