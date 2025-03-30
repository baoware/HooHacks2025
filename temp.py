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
    On a short button press, open a new terminal and run the obstacle detection program.
    The terminal remains open after the program completes.
    """
    global process
    if process is None:
        print("Starting the obstacle detection program in a new terminal...")
        # Construct absolute paths for clarity.
        script_path = os.path.join(os.getcwd(), "yolov5", "yolov5_obstacle_detection3.py")
        weights_path = os.path.join(os.getcwd(), "yolov5n.pt")
        # Build the command to run in a new terminal window.
        # Using lxterminal with a bash command that ends with 'exec bash' to keep the terminal open.
        cmd = [
            "lxterminal",
            "-e",
            f"bash -c '{sys.executable} {script_path} --weights {weights_path} --source 0 --img 640; exec bash'"
        ]
        process = subprocess.Popen(cmd, env=os.environ.copy())
    else:
        print("Program is already running. Doing nothing.")

def long_press_callback():
    """
    On a long button press, simulate a KeyboardInterrupt.
    """
    print("Long press detected. Raising KeyboardInterrupt!")
    raise KeyboardInterrupt

# Bind the button events.
button.when_pressed = toggle_program
button.when_held = long_press_callback

def main():
    print("System ready:")
    print("- Press the button briefly to open a new terminal and run the obstacle detection program (if not already running).")
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
                pass  # Process may already have ended.
            process = None
        # Restart this button script
        os.execv(sys.executable, [sys.executable] + sys.argv)

if __name__ == '__main__':
    main()
