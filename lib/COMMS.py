import RPi.GPIO as GPIO
import time

# -------------------------
# GPIO Pin Definitions (using BCM numbering)
# -------------------------

CS_PIN         = 26   # Output: Chip Select for transmission
CLOCK_PIN      = 19   # Output: Clock for parallel transmission
DATA_X_PIN     = 13   # Output: Data for X coordinate
DATA_Y_PIN     = 6  # Output: Data for Y coordinate
DATA_ANGLE_PIN = 5   # Output: Data for Angle

REQUEST_PIN    = 21   # Input: Triggered when pulled LOW
# -------------------------
# Timing Configuration
# -------------------------

BIT_DELAY = 0.03  # Delay (in seconds) per clock transition

# -------------------------
# Setup GPIO
# -------------------------
GPIO.setmode(GPIO.BCM)

# Setup all transmission pins as outputs and initialize them to LOW
for pin in [CS_PIN, CLOCK_PIN, DATA_X_PIN, DATA_Y_PIN, DATA_ANGLE_PIN]:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

# Set CS and CLOCK to idle HIGH
GPIO.output(CS_PIN, GPIO.HIGH)
GPIO.output(CLOCK_PIN, GPIO.HIGH)

def send_to_vex_brain(x, y, angle):
    """
    Transmits x, y, and angle (angle in tenths of a degree) as 12-bit words in parallel.
    x and y are signed 12-bit (two's complement), and angle is an unsigned 12-bit value.
    """
    # Ensure the yaw angle is non-negative (if negative, adjust by adding 360°)
    if angle < 0:
        angle += 360.0

    # Convert x, y to integers; convert yaw to tenths of a degree (as an integer)
    x = int(round(x))
    y = int(round(y))
    angle = int(round(angle * 10))

    # Convert values to 12-bit representations
    ux = x & 0xFFF
    uy = y & 0xFFF
    uangle = angle & 0xFFF  # angle is already scaled (tenths of a degree)

    # Begin transmission: Pull CS LOW
    GPIO.output(CS_PIN, GPIO.LOW)
    time.sleep(BIT_DELAY)

    # Send 12 bits (MSB first)
    for bit in range(11, -1, -1):
        bit_x     = (ux >> bit) & 0x01
        bit_y     = (uy >> bit) & 0x01
        bit_angle = (uangle >> bit) & 0x01

        # Set the data pins for this bit
        GPIO.output(DATA_X_PIN, bit_x)
        GPIO.output(DATA_Y_PIN, bit_y)
        GPIO.output(DATA_ANGLE_PIN, bit_angle)

        # Pulse the clock
        GPIO.output(CLOCK_PIN, GPIO.LOW)
        time.sleep(BIT_DELAY)
        GPIO.output(CLOCK_PIN, GPIO.HIGH)
        time.sleep(BIT_DELAY)

    # End transmission: release CS and clear data pins
    GPIO.output(CS_PIN, GPIO.HIGH)
    GPIO.output(DATA_X_PIN, GPIO.LOW)
    GPIO.output(DATA_Y_PIN, GPIO.LOW)
    GPIO.output(DATA_ANGLE_PIN, GPIO.LOW)

    print (f'SEND X = {x}, Y = {y}, Angle = {angle}')
    
    
if __name__ == '__main__':

    import RPi.GPIO as GPIO

    # Setup the request pin (if you have an internal pull-up, you can use PUD_UP)
    GPIO.setup(REQUEST_PIN, GPIO.IN, pull_up_down=GPIO.PUD_OFF)

    # Initialize simulated position values.
    current_x = -2000  # starting X position (mm)
    current_y = -2000  # starting Y position (mm)
    current_angle = 0  # starting angle in tenths of a degree

    def handle_request(channel):
        """
        Called when the request pin goes LOW.
        Computes the position and angle (here, simulated) and transmits them.
        """
        # For demonstration, we simulate position values.
        # In a real application, you would read sensors or compute the actual robot position.
        # Here, we simply cycle through some example values.
        # The x and y range from -2000 to 2000; angle is in tenths of a degree (0 to 3600).
        global current_x, current_y, current_angle

        # Calculate (or update) the simulated position.
        current_x += 50
        if current_x > 2000:
            current_x = -2000

        current_y += 30
        if current_y > 2000:
            current_y = -2000

        current_angle += 5
        if current_angle >= 360:
            current_angle = 0

        # Transmit the computed values in parallel.
        send_to_vex_brain(current_x, current_y, current_angle)

    # -------------------------
    # Set Up Event Detection on Request Pin
    # -------------------------
    # When the request pin goes LOW (rising edge when coming back to HIGH after being pulled LOW),
    # the callback function is triggered.
    GPIO.add_event_detect(REQUEST_PIN, GPIO.RISING, callback=handle_request)

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
