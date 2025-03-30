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
    On a short button press, open a new terminal, activate the current virtual environment,
    and run the obstacle detection program with the specific parameters.
    The command is:
      python3 yolov5/yolov5_obstacle_detection3.py --weights yolov5n.pt --source 0 --img 640
    The terminal will remain open after the command finishes.
    """
    global process
    if process is None:
        print("Starting the obstacle detection program in a new terminal...")

        # Determine the path to your virtual environment's activation script.
        # sys.executable should be something like '/path/to/venv/bin/python3'
        venv_bin_dir = os.path.dirname(sys.executable)
        activate_script = os.path.join(venv_bin_dir, "activate")

        # Construct absolute paths for clarity.
        script_path = os.path.join(os.getcwd(), "yolov5", "yolov5_obstacle_detection3.py")
        weights_path = os.path.join(os.getcwd(), "yolov5n.pt")

        # Build the specific command you want to run.
        # It sources the virtual environment activation script, then runs the command,
        # and finally executes bash to keep the terminal open.
        specific_command = (
            f"source {activate_script}; "
            f"python3 {script_path} --weights {weights_path} --source 0 --img 640; "
            "exec bash"
        )

        # Build the command for lxterminal.
        cmd = [
            "lxterminal",
            "-e",
            f"bash -c '{specific_command}'"
        ]

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
    print("- Press the button briefly to open a new terminal, activate the venv, and run:")
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
                pass  # Process may have already ended.
            process = None
        # Restart the current script (this reinitializes the button control)
        os.execv(sys.executable, [sys.executable] + sys.argv)

if __name__ == '__main__':
    main()
