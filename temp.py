from gpiozero import Button
import subprocess
import time
import os
import signal
import sys

# Define the GPIO pin for the button (using GPIO 17 as an example)
button = Button(17, hold_time=3)

# Global variable to hold the process (if any)
process = None

def toggle_program():
    """
    On a short button press, start the obstacle detection program if it's not already running.
    """
    global process
    if process is None:
        print("Starting the obstacle detection program...")
        # Run the command to start the program.
        process = subprocess.Popen([
            "python3", "yolov5/yolov5_obstacle_detection3.py",
            "--weights", "yolov5n.pt",
            "--source", "0",
            "--img", "640"
        ])
    else:
        print("Program is already running. Doing nothing.")

def long_press_callback():
    """
    On a long button press, simulate a KeyboardInterrupt.
    """
    print("Long press detected. Raising KeyboardInterrupt!")
    raise KeyboardInterrupt

# Bind the short press and long press events to their functions.
button.when_pressed = toggle_program
button.when_held = long_press_callback

def main():
    print("System ready:")
    print("- Press the button briefly to start the obstacle detection program (if not already running).")
    print("- Hold the button for 3 seconds to simulate a KeyboardInterrupt and restart this button program.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught. Terminating any running program and restarting...")
        # If a child process is running, terminate it.
        global process
        if process is not None:
            try:
                os.kill(process.pid, signal.SIGTERM)
            except OSError:
                pass  # Process might already be terminated.
            process = None
        # Restart the script.
        os.execv(sys.executable, [sys.executable] + sys.argv)

if __name__ == '__main__':
    main()
