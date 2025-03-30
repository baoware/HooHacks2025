from gpiozero import Button
import subprocess
import time
import os
import signal
import sys

# Use GPIO 17 with a hold time of 3 seconds for long-press detection
button = Button(17, hold_time=3)

# Global variable to track the running process (if any)
process = None

# Define the path to your virtual environment's Python interpreter
venv_path = "/home/John/Desktop/TWK2025/Hoohacks2025/venv"
venv_python = os.path.join(venv_path, "bin", "python3")

def toggle_program():
    """
    On a short button press:
    - If the program is not running, start it.
    - If the program is running, do nothing.
    """
    global process
    if process is None:
        print("Starting the obstacle detection program...")
        # Construct absolute paths for clarity
        script_path = os.path.join(os.getcwd(), "yolov5", "yolov5_obstacle_detection3.py")
        weights_path = os.path.join(os.getcwd(), "yolov5n.pt")
        
        # Build the command using the venv's python
        cmd = [
            venv_python,
            script_path,
            "--weights", weights_path,
            "--source", "0",
            "--img", "640"
        ]
        
        # Start the program as a background process
        process = subprocess.Popen(cmd, env=os.environ.copy())
    else:
        print("Program is already running. To stop it, hold the button for 3 seconds.")

def long_press_callback():
    """
    On a long button press (3 seconds), stop the running program if it is running.
    """
    global process
    if process is not None:
        print("Long press detected. Stopping the program...")
        try:
            os.kill(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except Exception as e:
            print("Error stopping process:", e)
        process = None
    else:
        print("No program is running to stop.")

def main():
    print("System ready:")
    print("- Press the button briefly to start the obstacle detection program (if not already running).")
    print("- Hold the button for 3 seconds to stop the running program.")
    while True:
        time.sleep(1)

# Bind the button events
button.when_pressed = toggle_program
button.when_held = long_press_callback

if __name__ == '__main__':
    main()
