#!/usr/bin/env python3
import RPi.GPIO as GPIO
import time
from lib.COMMS import send_to_vex_brain, kill_camera
from lib.Get_Position_from_Markers import get_position_from_markers

# -------------------------
# GPIO Pin Definition
# -------------------------
REQUEST_PIN = 17   # Input: Triggered when pulled LOW

def handle_request(channel):
    """
    Called when the request pin is triggered.
    Retrieves the camera's position and yaw using get_position_from_markers(),
    converts the values to the appropriate format, and transmits them.
    """
    pos, yaw = get_position_from_markers()
    if pos is not None:
        print(f"\n[RESULT] Camera Position: {pos}")
        print(f"[RESULT] Camera Yaw: {yaw:.1f} deg")
        # Convert x and y to integers and yaw to tenths of a degree.
        x = int(round(pos[0]))
        y = int(round(pos[1]))
        angle = int(round(yaw * 10))
        send_to_vex_brain(x, y, angle)
    else:
        print("Failed to determine camera pose.")

if __name__ == '__main__':
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(REQUEST_PIN, GPIO.IN, pull_up_down=GPIO.PUD_OFF)

    # Set up event detection on the request pin.
    GPIO.add_event_detect(REQUEST_PIN, GPIO.RISING, callback=handle_request, bouncetime=200)

    try:
        print("Waiting for request signal...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        GPIO.cleanup()
        kill_camera()
