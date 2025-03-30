import serial
import time

# Adjust the device path if necessary (e.g., '/dev/ttyACM0' on Linux/Mac or 'COM3' on Windows)
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 9600

def send_command(command):
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2) as ser:
            # Allow the Arduino to reset if necessary
            time.sleep(1)
            print(f"Sending: {command}")
            ser.write((command + "\n").encode('utf-8'))
            # Optionally, you can read a response:
            # response = ser.readline().decode('utf-8').strip()
            # print("Response:", response)
    except serial.SerialException as e:
        print("Error:", e)

def alert_high():
    send_command("High")

def alert_medium_far():
    send_command("Medium Far")
    
def alert_medium_close():
    send_command("Medium Close")
    
def alert_low_far():
    send_command("Low Far")

def alert_low_close():
    send_command("Low Close")


