#!/usr/bin/env python3
import cv2
import numpy as np
import time
import threading
from flask import Flask, Response, render_template_string
from lib.Get_Position_from_Markers import get_position_from_markers, make_map_image

UPDATE_TIME_SECONDS = 0.2

# -----------------------------------------------------------
# Global Variables for Sharing Data
# -----------------------------------------------------------
latest_frame = None      # Annotated camera frame (BGR image)
latest_cam_pos = None    # Robot position as a NumPy array [x, y, z] (in mm)
latest_yaw = None        # Robot yaw (in degrees)
data_lock = threading.Lock()

# -----------------------------------------------------------
# Flask App Setup with Bootstrap Template
# -----------------------------------------------------------
app = Flask(__name__)

html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Robot Location & Camera Stream</title>
  <!-- Bootstrap CSS -->
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
          <div class="card-header text-center">Camera Stream</div>
          <div class="card-body p-0">
            <img src="{{ url_for('video_feed') }}" class="stream-img" alt="Camera Stream">
          </div>
        </div>
      </div>
      <div class="col-md-6 mb-3">
        <div class="card">
          <div class="card-header text-center">2D Map</div>
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
      // Refresh pose info every second
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
        time.sleep(UPDATE_TIME_SECONDS)

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_map_stream():

    global latest_cam_pos, latest_yaw
    
    while True:
        with data_lock:
            if latest_cam_pos is None or latest_yaw is None:
                # Create blank map if no pose data available.
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
    with data_lock:
        if latest_cam_pos is not None and latest_yaw is not None:
            info = f"Position: X={latest_cam_pos[0]:.1f}, Y={latest_cam_pos[1]:.1f}, Z={latest_cam_pos[2]:.1f}; Yaw: {latest_yaw:.1f} deg"
        else:
            info = "No pose data available."
    return info

def update_loop():
    """
    Continuously captures frames, processes them using get_position_from_markers(),
    and updates global variables.
    """
    global latest_frame, latest_cam_pos, latest_yaw
    while True:
        pos, yaw, frame = get_position_from_markers()
        with data_lock:
            latest_frame = frame.copy()
            latest_cam_pos = pos
            latest_yaw = yaw
        time.sleep(UPDATE_TIME_SECONDS)

if __name__ == "__main__":
    update_thread = threading.Thread(target=update_loop, daemon=True)
    update_thread.start()
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        pass
