import serial
import time

# Adjust the device path if necessary (e.g., /dev/ttyACM0)
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 9600

def send_command(command):
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2) as ser:
            # Give the Arduino a moment to reset if needed
            time.sleep(2)
            print(f"Sending: {command}")
            ser.write((command + "\n").encode('utf-8'))
            # Optionally, you can read a response:
            # response = ser.readline().decode('utf-8').strip()
            # print("Response:", response)
    except serial.SerialException as e:
        print("Error:", e)

def main():
    # Test each alert command with a delay between each test
    commands = ["ALERT HIGH", "ALERT MEDIUM", "ALERT LOW"]
    
    for cmd in commands:
        send_command(cmd)
        # Wait enough time for the alert sequence to complete on the Arduino.
        time.sleep(7)

if __name__ == '__main__':
    main()
