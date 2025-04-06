import RPi.GPIO as GPIO
import time
import requests

url = "http://10.15.171.6:5000/robot1"

# -------------------------
# GPIO Pin Definitions (using BCM numbering)
# -------------------------

REQUEST_PIN    = 17   # Input: Triggered when pulled LOW
CS_PIN         = 27   # Output: Chip Select for transmission
CLOCK_PIN      = 22   # Output: Clock for parallel transmission
DATA_X_PIN     = 23   # Output: Data for X coordinate
DATA_Y_PIN     = 24   # Output: Data for Y coordinate
DATA_ANGLE_PIN = 25   # Output: Data for Angle

# -------------------------
# Timing Configuration
# -------------------------
BIT_DELAY = 0.03  # Delay (in seconds) per clock transition

# -------------------------
# Setup GPIO
# -------------------------
GPIO.setmode(GPIO.BCM)
# Setup the request pin (if you have an internal pull-up, you can use PUD_UP)
GPIO.setup(REQUEST_PIN, GPIO.IN, pull_up_down=GPIO.PUD_OFF)

# Setup all transmission pins as outputs and initialize them to LOW
for pin in [CS_PIN, CLOCK_PIN, DATA_X_PIN, DATA_Y_PIN, DATA_ANGLE_PIN]:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

# Set CS and CLOCK to idle HIGH
GPIO.output(CS_PIN, GPIO.HIGH)
GPIO.output(CLOCK_PIN, GPIO.HIGH)

def send_parallel_data(x, y, angle):
    """
    Transmits x, y, and angle (angle in tenths of a degree) as 12-bit words in parallel.
    x and y are signed 12-bit (two's complement), and angle is an unsigned 12-bit value.
    """
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

    print(f"Transmitted: X={x}, Y={y}, Angle={angle/10.0}°")

def handle_request(channel):
    """
    Triggered when the request pin is pulled LOW.
    Makes an HTTP GET request to the specified URL,
    processes the returned marker data, ensures the angle is non-negative,
    and transmits the position and angle.
    """
    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            print("Error: Received status code", response.status_code)
            return

        data = response.json()
        # Expected JSON structure: {"id":1, "global_position": [x, y, z], "yaw": yaw_deg}
        x = data["global_position"][0]
        y = data["global_position"][1]
        yaw = data["yaw"]

        # Ensure the yaw angle is non-negative (if negative, adjust by adding 360°)
        if yaw < 0:
            yaw += 360.0

        # Convert x, y to integers; convert yaw to tenths of a degree (as an integer)
        x = int(round(x))
        y = int(round(y))
        angle = int(round(yaw * 10))

        send_parallel_data(x, y, angle)
    except Exception as e:
        print("Error in handle_request:", e)

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
