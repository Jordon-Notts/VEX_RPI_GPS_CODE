#!/usr/bin/env python3
import cv2
import numpy as np
import time
import threading
from flask import Flask, Response, render_template
from lib.Get_Position_from_Markers import get_position_from_markers, make_map_image, kill_camera
from lib.COMMS import send_to_vex_brain
import RPi.GPIO as GPIO

UPDATE_TIME_SECONDS = 0.2

app = Flask(__name__)

# Global Variables for Sharing Data
latest_frame = None      # Latest annotated camera frame (BGR image)
latest_cam_pos = None    # Robot position as a NumPy array [x, y, z] (in mm)
latest_yaw = None        # Robot yaw (in degrees)
data_lock = threading.Lock()

@app.route('/')
def index():
    global latest_cam_pos, latest_yaw
    with data_lock:
        if latest_cam_pos is not None and latest_yaw is not None:
            pose_info = (f"Position: X={latest_cam_pos[0]:.1f}, Y={latest_cam_pos[1]:.1f}, "
                         f"Z={latest_cam_pos[2]:.1f}; Yaw: {latest_yaw:.1f} deg")
        else:
            pose_info = "No pose data available."
    return render_template("index.html", pose_info=pose_info)

def generate_video_stream():
    """
    Captures a fresh frame via get_position_from_markers(), updates global variables,
    and yields the annotated frame as a JPEG.
    """
    global latest_frame, latest_cam_pos, latest_yaw
    while True:
        pos, yaw, frame = get_position_from_markers()
        with data_lock:
            latest_frame = frame.copy()
            latest_cam_pos = pos
            latest_yaw = yaw
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(UPDATE_TIME_SECONDS)

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_map_stream():
    """
    Uses the latest global pose values to generate a 2D map image,
    then yields the JPEG-encoded map image.
    """
    global latest_cam_pos, latest_yaw
    while True:
        with data_lock:
            if latest_cam_pos is None or latest_yaw is None:
                map_img = np.ones((500, 500, 3), dtype=np.uint8) * 255
            else:
                map_img = make_map_image(latest_cam_pos, latest_yaw)
        ret, buffer = cv2.imencode('.jpg', map_img)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(UPDATE_TIME_SECONDS)

@app.route('/map_feed')
def map_feed():
    return Response(generate_map_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pose_info')
def pose_info():
    global latest_cam_pos, latest_yaw
    with data_lock:
        if latest_cam_pos is not None and latest_yaw is not None:
            info = (f"Position: X={latest_cam_pos[0]:.1f}, Y={latest_cam_pos[1]:.1f}, "
                    f"Z={latest_cam_pos[2]:.1f}; Yaw: {latest_yaw:.1f} deg")
        else:
            info = "No pose data available."
    return info

# -----------------------------------------------------------
# GPIO Setup for VEX Brain Request
# -----------------------------------------------------------
REQUEST_PIN = 17  # Input: Triggered by the VEX brain

GPIO.setmode(GPIO.BCM)
GPIO.setup(REQUEST_PIN, GPIO.IN, pull_up_down=GPIO.PUD_OFF)

def handle_request(channel):
    """
    Called when the request pin is triggered.
    Captures the current pose (on-demand) and sends X, Y, and yaw to the VEX brain.
    """
    try:
        pos, yaw, _ = get_position_from_markers()
        if pos is not None:
            print(f"\n[RESULT] Camera Position: {pos}")
            print(f"[RESULT] Camera Yaw: {yaw:.1f} deg")
            send_to_vex_brain(pos[0], pos[1], yaw)
        else:
            print("Failed to determine camera pose for transmission.")
    except Exception as e:
        print("Error in handle_request:", e)

GPIO.add_event_detect(REQUEST_PIN, GPIO.RISING, callback=handle_request)

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()
        kill_camera()
