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
    On a short button press, open a new terminal and run the specified obstacle detection command.
    If the process is already running, do nothing.
    """
    global process
    if process is None:
        print("Starting the obstacle detection program in a new terminal...")
        # The specific command to run:
        specific_command = "python3 yolov5/yolov5_obstacle_detection3.py --weights yolov5n.pt --source 0 --img 640"
        # Build the command for lxterminal.
        # 'bash -c' executes the specific command and then 'exec bash' keeps the terminal open.
        cmd = [
            "lxterminal",
            "-e",
            f"bash -c '{specific_command}; exec bash'"
        ]
        process = subprocess.Popen(cmd, env=os.environ.copy())
    else:
        print("Program is already running. Doing nothing.")

def long_press_callback():
    """
    On a long button press, simulate a KeyboardInterrupt.
    This will terminate the running process (if any) and restart the button program.
    """
    print("Long press detected. Raising KeyboardInterrupt!")
    raise KeyboardInterrupt

# Bind the short press and long press events to their functions.
button.when_pressed = toggle_program
button.when_held = long_press_callback

def main():
    print("System ready:")
    print("- Press the button briefly to open a new terminal and run the obstacle detection program.")
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
                pass  # The process might have already terminated.
            process = None
        # Restart the current script
        os.execv(sys.executable, [sys.executable] + sys.argv)

if __name__ == '__main__':
    main()
