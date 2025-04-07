#!/usr/bin/env python3
import cv2
import numpy as np
import json
import os
import time
import math
import threading
from datetime import datetime
from flask import Flask, Response, render_template_string, url_for
from picamera2 import Picamera2

# -----------------------------------------------------------
# Configuration / Paths
# -----------------------------------------------------------
CALIBRATION_FILE = "camera_calibration.json"
MARKER_DB_FILE = "aruco_marker_positions.json"
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
# Load Marker Database
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
# Global Variables and Lock
# -----------------------------------------------------------
latest_frame = None      # Annotated camera frame (BGR image)
latest_cam_pos = None    # Robot position as a NumPy array [x, y, z] (in mm)
latest_yaw = None        # Yaw (heading) in degrees
data_lock = threading.Lock()

# -----------------------------------------------------------
# Helper Functions
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

def update_loop():
    """
    Continuously captures frames, detects markers, computes pose,
    and updates global variables.
    """
    global latest_frame, latest_cam_pos, latest_yaw
    while True:
        frame = picam2.capture_array()
        # (Optional) Mirror the frame:
        # frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, _ = detector.detectMarkers(gray)
        
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            all_obj_points = []
            all_img_points = []
            for i, marker in enumerate(ids.flatten()):
                marker_id = str(marker)
                if marker_id in marker_db:
                    pts = np.int32(corners[i][0])
                    # Draw circles at corners.
                    for pt in pts:
                        cv2.circle(frame, tuple(pt), 5, (0,255,0), -1)
                    center = np.mean(pts, axis=0).astype(int)
                    cv2.putText(frame, f"ID {marker_id}", (center[0]-10, center[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    # Get 3D marker corners.
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
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 50)
                    R, _ = cv2.Rodrigues(rvec)
                    cam_pos = -R.T.dot(tvec)
                    forward_vector = R.T.dot(np.array([[0], [0], [1]], dtype=np.float32))
                    yaw = np.degrees(np.arctan2(forward_vector[1,0], forward_vector[0,0]))
                    if yaw < 0:
                        yaw += 360
                    with data_lock:
                        latest_cam_pos = cam_pos.flatten()
                        latest_yaw = yaw
        else:
            with data_lock:
                latest_cam_pos = None
                latest_yaw = None
        with data_lock:
            latest_frame = frame.copy()
        time.sleep(0.1)

# Start the background update thread.
update_thread = threading.Thread(target=update_loop, daemon=True)
update_thread.start()

# -----------------------------------------------------------
# Flask Web Server Setup
# -----------------------------------------------------------
app = Flask(__name__)

html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Robot Location & Camera Stream</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body { background-color: #f8f9fa; }
    .stream-img { width: 100%; height: auto; }
  </style>
</head>
<body>
  <div class="container my-4">
    <h1 class="text-center mb-4">Robot Location & Camera Stream</h1>
    <div class="row">
      <div class="col-md-6 mb-3">
        <div class="card">
          <div class="card-header text-center">
            Camera Stream
          </div>
          <div class="card-body p-0">
            <img src="{{ url_for('video_feed') }}" class="stream-img" alt="Camera Stream">
          </div>
        </div>
      </div>
      <div class="col-md-6 mb-3">
        <div class="card">
          <div class="card-header text-center">
            2D Map
          </div>
          <div class="card-body p-0">
            <img src="{{ url_for('map_feed') }}" class="stream-img" alt="Map Stream">
          </div>
        </div>
      </div>
    </div>
    <div class="row mt-4">
      <div class="col">
        <div class="alert alert-info text-center" role="alert" id="pose_info">
          {{ pose_info }}
        </div>
      </div>
    </div>
    <script>
      setInterval(function(){
          fetch("{{ url_for('pose_info') }}")
          .then(response => response.text())
          .then(text => { document.getElementById("pose_info").innerText = text; });
      }, 1000);
    </script>
  </div>
</body>
</html>
"""

@app.route('/')
def index():
    with data_lock:
        if latest_cam_pos is not None and latest_yaw is not None:
            pose_info = f"Position: X={latest_cam_pos[0]:.1f}, Y={latest_cam_pos[1]:.1f}, Z={latest_cam_pos[2]:.1f}; Yaw: {latest_yaw:.1f} deg"
        else:
            pose_info = "No pose data available."
    return render_template_string(html_template, pose_info=pose_info)

def generate_video_stream():
    global latest_frame
    while True:
        with data_lock:
            if latest_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_map_stream():
    """
    Creates a live 2D map image with:
      - A white background with grid lines.
      - A field boundary of 3657 mm x 3657 mm, centered on the map.
      - The robot represented as a 400x400 mm square (rotated to match its heading) with an arrow.
    """
    map_size = 500  # pixels (map image size)
    map_img = np.ones((map_size, map_size, 3), dtype=np.uint8) * 255
    map_center = (map_size // 2, map_size // 2)
    scale = 0.1  # Scale: 1 mm = 0.1 pixel (adjust as needed)

    # Draw grid lines.
    for i in range(0, map_size, 50):
        cv2.line(map_img, (i, 0), (i, map_size), (200, 200, 200), 1)
        cv2.line(map_img, (0, i), (map_size, i), (200, 200, 200), 1)

    # Draw field boundary (3657 mm x 3657 mm).
    field_size_mm = 3657
    half_field_pixels = int((field_size_mm * scale) / 2)
    top_left_field = (map_center[0] - half_field_pixels, map_center[1] - half_field_pixels)
    bottom_right_field = (map_center[0] + half_field_pixels, map_center[1] + half_field_pixels)
    cv2.rectangle(map_img, top_left_field, bottom_right_field, (0, 0, 0), 2)

    # Draw the robot as a 400x400 mm square that rotates with its heading.
    # Convert 400 mm to pixels: 400 * scale.
    robot_size_mm = 400.0
    robot_size_px = robot_size_mm * scale  # e.g. 400 * 0.1 = 40 pixels
    half_robot_mm = robot_size_mm / 2.0  # 200 mm

    # Define the robot square in its local coordinate system (in mm), centered at (0,0)
    square_local = np.array([
        [-half_robot_mm, -half_robot_mm],
        [ half_robot_mm, -half_robot_mm],
        [ half_robot_mm,  half_robot_mm],
        [-half_robot_mm,  half_robot_mm]
    ], dtype=np.float32)

    # Use a lock to safely access the shared latest_cam_pos and latest_yaw.
    with data_lock:
        if latest_cam_pos is not None and latest_yaw is not None:
            # Convert robot position from mm to map pixels.
            robot_x = int(map_center[0] + latest_cam_pos[0] * scale)
            robot_y = int(map_center[1] - latest_cam_pos[1] * scale)

            # Create a 2D rotation matrix from the yaw angle (in radians).
            yaw_rad = math.radians(latest_yaw)
            R2 = np.array([
                [math.cos(yaw_rad), -math.sin(yaw_rad)],
                [math.sin(yaw_rad),  math.cos(yaw_rad)]
            ], dtype=np.float32)

            # Rotate the local square vertices.
            rotated_square = np.dot(square_local, R2.T)
            # Scale from mm to pixels.
            rotated_square_px = rotated_square * scale
            # Translate the square to the robot's map position.
            square_map = rotated_square_px + np.array([robot_x, robot_y])
            square_map = square_map.astype(np.int32)

            # Draw the robot square (using blue color).
            cv2.polylines(map_img, [square_map.reshape((-1, 1, 2))], isClosed=True, color=(255, 0, 0), thickness=2)

            # Draw an arrow from the center of the square to indicate heading.
            arrow_length_px = 20  # length of arrow in pixels
            arrow_end = (int(robot_x + arrow_length_px * math.cos(yaw_rad)),
                         int(robot_y - arrow_length_px * math.sin(yaw_rad)))
            cv2.arrowedLine(map_img, (robot_x, robot_y), arrow_end, (255, 0, 0), 2)
            cv2.putText(map_img, "Robot", (robot_x - 30, robot_y - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    ret, buffer = cv2.imencode('.jpg', map_img)
    if ret:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    time.sleep(0.5)

@app.route('/map_feed')
def map_feed():
    return Response(generate_map_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pose_info')
def pose_info():
    with data_lock:
        if latest_cam_pos is not None and latest_yaw is not None:
            info = f"Position: X={latest_cam_pos[0]:.1f}, Y={latest_cam_pos[1]:.1f}, Z={latest_cam_pos[2]:.1f}; Yaw: {latest_yaw:.1f} deg"
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
        picam2.stop()



        
