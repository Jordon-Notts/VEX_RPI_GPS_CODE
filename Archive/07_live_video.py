#!/usr/bin/env python3
import cv2
import numpy as np
import json
import os
import time
import math
from datetime import datetime
from flask import Flask, Response, render_template_string
from picamera2 import Picamera2

# -----------------------------------------------------------
# Configuration / Paths
# -----------------------------------------------------------
CALIBRATION_FILE = "camera_calibration.json"
MARKER_DB_FILE = "aruco_marker_positions.json"
# Create output folder if needed (not used for streaming but useful for saving images)
OUTPUT_FOLDER = "output_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------------------------------------
# Load Camera Calibration Data
# -----------------------------------------------------------
if not os.path.exists(CALIBRATION_FILE):
    raise FileNotFoundError(f"Missing {CALIBRATION_FILE}")
with open(CALIBRATION_FILE, "r") as f:
    calib_data = json.load(f)
camera_matrix = np.array(calib_data["camera_matrix"], dtype=np.float32)
dist_coeffs   = np.array(calib_data["distortion_coeffs"], dtype=np.float32)

# -----------------------------------------------------------
# Load Marker Database (with known 3D marker corners)
# -----------------------------------------------------------
if not os.path.exists(MARKER_DB_FILE):
    raise FileNotFoundError(f"Missing {MARKER_DB_FILE}")
with open(MARKER_DB_FILE, "r") as f:
    marker_db = json.load(f)

# -----------------------------------------------------------
# Setup ArUco Detector
# -----------------------------------------------------------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

# -----------------------------------------------------------
# Initialize PiCamera2 (once for quick capture)
# -----------------------------------------------------------
picam2 = Picamera2()
config = picam2.create_still_configuration(main={"size": (1280, 720), "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(1)  # Allow camera to warm up

# -----------------------------------------------------------
# Helper Function: Convert Rotation Matrix to Euler Angles
# -----------------------------------------------------------
def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        roll  = math.atan2(R[2,1], R[2,2])
        pitch = math.atan2(-R[2,0], sy)
        yaw   = math.atan2(R[1,0], R[0,0])
    else:
        roll  = math.atan2(-R[1,2], R[1,1])
        pitch = math.atan2(-R[2,0], sy)
        yaw   = 0
    return np.array([roll, pitch, yaw])

# -----------------------------------------------------------
# Video Stream Generator
# -----------------------------------------------------------
def generate_frames():
    while True:
        frame = picam2.capture_array()
        # Optionally mirror the frame:
        # frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect markers.
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, _ = detector.detectMarkers(gray)
        
        if ids is not None and len(ids) > 0:
            # Draw marker outlines and labels.
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            all_obj_points = []
            all_img_points = []
            for i, marker in enumerate(ids.flatten()):
                marker_id = str(marker)
                if marker_id in marker_db:
                    pts = np.int32(corners[i][0])
                    for pt in pts:
                        cv2.circle(frame, tuple(pt), 5, (0,255,0), -1)
                    center = np.mean(pts, axis=0).astype(int)
                    cv2.putText(frame, f"ID {marker_id}", (center[0]-10, center[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    # Prepare correspondence for pose.
                    data = marker_db[marker_id]["corner_points"]
                    corners_world = np.array([
                        [data["top_left"]["x"],     data["top_left"]["y"],     data["top_left"]["z"]],
                        [data["top_right"]["x"],    data["top_right"]["y"],    data["top_right"]["z"]],
                        [data["bottom_right"]["x"], data["bottom_right"]["y"], data["bottom_right"]["z"]],
                        [data["bottom_left"]["x"],  data["bottom_left"]["y"],  data["bottom_left"]["z"]]
                    ], dtype=np.float32)
                    all_obj_points.append(corners_world)
                    all_img_points.append(corners[i][0])
            
            if len(all_obj_points) > 0:
                all_obj_points = np.vstack(all_obj_points)
                all_img_points = np.vstack(all_img_points)
                ret, rvec, tvec = cv2.solvePnP(all_obj_points, all_img_points,
                                               camera_matrix, dist_coeffs,
                                               flags=cv2.SOLVEPNP_ITERATIVE)
                if ret:
                    # Draw the global coordinate axes (trident) at the world origin.
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 50)
                    # Compute camera position.
                    R, _ = cv2.Rodrigues(rvec)
                    cam_pos = -R.T.dot(tvec)
                    # Compute yaw (heading) using optical axis [0,0,1].
                    forward_vector = R.T.dot(np.array([[0], [0], [1]], dtype=np.float32))
                    yaw = np.degrees(np.arctan2(forward_vector[1,0], forward_vector[0,0]))
                    if yaw < 0:
                        yaw += 360
                    pos_text = f"Pos: X={cam_pos[0,0]:.1f}, Y={cam_pos[1,0]:.1f}, Z={cam_pos[2,0]:.1f}"
                    yaw_text = f"Yaw: {yaw:.1f} deg"
                    cv2.putText(frame, pos_text, (20, frame.shape[0]-40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                    cv2.putText(frame, yaw_text, (20, frame.shape[0]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                    
        ret2, buffer = cv2.imencode('.jpg', frame)
        if not ret2:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.1)

# -----------------------------------------------------------
# Flask Web Server Setup
# -----------------------------------------------------------
app = Flask(__name__)

HTML_PAGE = """
<html>
  <head>
    <title>Robot Location & Camera Stream</title>
    <style>
      body { text-align: center; background-color: #f0f0f0; }
      h1 { color: #333; }
      #video-stream { width: 80%; }
    </style>
  </head>
  <body>
    <h1>Robot Location & Camera Stream</h1>
    <img id="video-stream" src="{{ url_for('video_feed') }}">
  </body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
if __name__ == '__main__':
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
