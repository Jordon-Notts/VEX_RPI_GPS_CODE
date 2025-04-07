#!/usr/bin/env python3
import cv2
import numpy as np
import json
import os
import time
import math
import threading
from flask import Flask, Response, render_template, url_for
from picamera2 import Picamera2
from lib.Get_Position_from_Markers import get_position_from_markers, make_map_image, kill_camera

UPDATE_TIME_SECONDS = 0.5

# -----------------------------------------------------------
# Global Variables for Sharing Data
# -----------------------------------------------------------
latest_frame = None      # Latest annotated camera frame (BGR image)
latest_x = None          # Latest X position (mm)
latest_y = None          # Latest Y position (mm)
latest_a = None          # Latest yaw (degrees)
data_lock = threading.Lock()

# -----------------------------------------------------------
# Flask App Setup with Bootstrap Template (HTML in templates folder)
# -----------------------------------------------------------
app = Flask(__name__)

@app.route('/')
def index():
    global latest_x, latest_y, latest_a
    if latest_x is not None and latest_y is not None and latest_a is not None:
        pose_info = f"Position: X={latest_x:.1f}, Y={latest_y:.1f}; Yaw: {latest_a:.1f} deg"
    else:
        pose_info = "No pose data available."
    return render_template("index.html", pose_info=pose_info)

def generate_video_stream():
    """
    Captures a fresh frame by calling get_position_from_markers(),
    updates the global variables (latest_x, latest_y, latest_a, latest_frame),
    and yields the JPEG-encoded annotated frame.
    """
    global latest_frame, latest_x, latest_y, latest_a
    while True:
        pos, yaw, frame = get_position_from_markers()
        with data_lock:
            latest_frame = frame.copy()
            if pos is not None and yaw is not None:
                latest_x = pos[0]
                latest_y = pos[1]
                latest_a = yaw
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
    Uses the latest global position values to generate a 2D map image.
    If the latest values are not available, a blank map is produced.
    """
    global latest_x, latest_y, latest_a
    while True:
        with data_lock:
            if latest_x is None or latest_y is None or latest_a is None:
                map_img = np.ones((500, 500, 3), dtype=np.uint8) * 255
            else:
                pos = np.array([latest_x, latest_y, 0])
                map_img = make_map_image(pos, latest_a)
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
    global latest_x, latest_y, latest_a
    if latest_x is not None and latest_y is not None and latest_a is not None:
        info = f"Position: X={latest_x:.1f}, Y={latest_y:.1f}; Yaw: {latest_a:.1f} deg"
    else:
        info = "No pose data available."
    return info

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        kill_camera()
