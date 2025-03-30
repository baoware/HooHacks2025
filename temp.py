from gpiozero import Button
import subprocess
import time
import os
import signal
import sys

# Define the GPIO pin for the button (using GPIO 17 as an example)
# hold_time=3 seconds means a long press triggers the long press callback.
button = Button(17, hold_time=3)

# Global variable to hold the process (if any)
process = None

def toggle_program():
    """
    This function is called on a short button press.
    If no process is running, it starts the program.
    If a process is already running, it stops it.
    """
    global process
    if process is None:
        print("Starting the program...")
        # Call your program here; for example, running a Python script.
        # Replace "your_program.py" with your actual script or command.
        process = subprocess.Popen(["python3", "your_program.py"])
    else:
        print("Stopping the program...")
        os.kill(process.pid, signal.SIGTERM)
        process = None

def long_press_callback():
    """
    This function is called on a long press.
    It raises a KeyboardInterrupt, which can be caught to shut down the program gracefully.
    """
    print("Long press detected. Raising KeyboardInterrupt!")
    raise KeyboardInterrupt

# Bind the toggle function to a short press and the long press callback to a long press.
button.when_pressed = toggle_program
button.when_held = long_press_callback

print("System ready:")
print("- Press the button briefly to start or stop the program.")
print("- Hold the button for 3 seconds to simulate a KeyboardInterrupt.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("KeyboardInterrupt caught. Exiting program gracefully.")
    sys.exit(0)
