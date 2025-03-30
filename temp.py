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
    This function is called on a short button press.
    If no process is running, it starts the program.
    If a process is already running, it does nothing.
    """
    global process
    if process is None:
        print("Starting the program...")
        # Replace "your_program.py" with your actual script or command.
        process = subprocess.Popen(["python3", "your_program.py"])
    else:
        print("Program is already running. Doing nothing.")

def long_press_callback():
    """
    This function is called on a long press.
    It raises a KeyboardInterrupt to shut down the program gracefully.
    """
    print("Long press detected. Raising KeyboardInterrupt!")
    raise KeyboardInterrupt

# Bind the toggle function to a short press and the long press callback to a long press.
button.when_pressed = toggle_program
button.when_held = long_press_callback

print("System ready:")
print("- Press the button briefly to start the program (if it's not already running).")
print("- Hold the button for 3 seconds to simulate a KeyboardInterrupt.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("KeyboardInterrupt caught. Exiting program gracefully.")
    sys.exit(0)
