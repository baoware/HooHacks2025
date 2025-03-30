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

# Define your virtual environment's root path.
venv_path = "/home/John/Desktop/TWK2025/Hoohacks2025/venv"
venv_python = os.path.join(venv_path, "bin", "python3")

def toggle_program():
    """
    On a short button press, run the obstacle detection program using the venv's Python.
    The specific command is:
      python3 yolov5/yolov5_obstacle_detection3.py --weights yolov5n.pt --source 0 --img 640
    The output will appear in this terminal.
    """
    global process
    if process is None:
        print("Starting the obstacle detection program in the current terminal...")

        # Construct absolute paths for clarity.
        script_path = os.path.join(os.getcwd(), "yolov5", "yolov5_obstacle_detection3.py")
        weights_path = os.path.join(os.getcwd(), "yolov5n.pt")

        # Build the command using the venv's Python interpreter.
        cmd = [
            venv_python,
            script_path,
            "--weights", weights_path,
            "--source", "0",
            "--img", "640"
        ]

        # Launch the process in the same terminal.
        process = subprocess.Popen(cmd, env=os.environ.copy())
    else:
        print("Program is already running. Doing nothing.")

def long_press_callback():
    """
    On a long button press, simulate a KeyboardInterrupt.
    This will terminate any running process and restart the button program.
    """
    print("Long press detected. Raising KeyboardInterrupt!")
    raise KeyboardInterrupt

# Bind the button events.
button.when_pressed = toggle_program
button.when_held = long_press_callback

def main():
    print("System ready:")
    print("- Press the button briefly to run:")
    print("  python3 yolov5/yolov5_obstacle_detection3.py --weights yolov5n.pt --source 0 --img 640")
    print("- Hold the button for 3 seconds to simulate a KeyboardInterrupt and restart the button program.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught. Terminating any running program and restarting...")
        global process
        if process is not None:
            try:
                os.kill(process.pid, signal.SIGTERM)
            except OSError:
                pass
            process = None
        # Restart the button script (keeping it in the current terminal)
        os.execv(sys.executable, [sys.executable] + sys.argv)

if __name__ == '__main__':
    main()
