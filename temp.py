import serial
import time

# Adjust the device path if necessary (e.g., '/dev/ttyACM0' on Linux/Mac or 'COM3' on Windows)
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 9600

def send_command(command):
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2) as ser:
            # Allow the Arduino to reset if necessary
            time.sleep(2)
            print(f"Sending: {command}")
            ser.write((command + "\n").encode('utf-8'))
            # Optionally, you can read a response:
            # response = ser.readline().decode('utf-8').strip()
            # print("Response:", response)
    except serial.SerialException as e:
        print("Error:", e)

def alert_high():
    """Send a high-level alert command."""
    send_command("ALERT HIGH")

def alert_medium():
    """Send a medium-level alert command."""
    send_command("ALERT MEDIUM")

def alert_low():
    """Send a low-level alert command."""
    send_command("ALERT LOW")

def main():
    # Call each alert function with a delay between them.
    alert_high()
    time.sleep(7)  # Wait enough time for the high alert sequence to complete
    alert_medium()
    time.sleep(7)  # Wait enough time for the medium alert sequence to complete
    alert_low()
    time.sleep(7)  # Wait enough time for the low alert sequence to complete

if __name__ == '__main__':
    main()
