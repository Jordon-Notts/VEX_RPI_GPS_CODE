import RPi.GPIO as GPIO
import time
from lib.COMMS import *
from lib.Get_Position_from_Server import *

URL = "http://yourserver:port/yourendpoint"  # Replace with your actual URL

def handle_request(channel):
    """
    Called when the request pin goes LOW.
    Computes the position and angle (here, simulated) and transmits them.
    """

    result = GET_position_from_server(URL)

    if result:

        current_x, current_y, current_angle = result
                    # Transmit the computed values in parallel.
        send_to_vex_brain(current_x, current_y, current_angle)

        print(f"Received Position: X={current_x}, Y={current_y}, Angle={current_angle/10.0:.1f} deg")
        
    else:
        print("Failed to get position from server.")


if __name__ == '__main__':

    import RPi.GPIO as GPIO

    REQUEST_PIN    = 17   # Input: Triggered when pulled LOW
    # Setup the request pin (if you have an internal pull-up, you can use PUD_UP)
    GPIO.setup(REQUEST_PIN, GPIO.IN, pull_up_down=GPIO.PUD_OFF)

    # -------------------------
    # Set Up Event Detection on Request Pin
    # -------------------------
    # When the request pin goes LOW (rising edge when coming back to HIGH after being pulled LOW),
    # the callback function is triggered.
    GPIO.add_event_detect(REQUEST_PIN, GPIO.RISING, callback=handle_request, bouncetime=200)

    # -------------------------
    # Main Loop
    # -------------------------
    try:
        print("Waiting for request signal...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        GPIO.cleanup()