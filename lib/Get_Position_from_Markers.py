#!/usr/bin/env python3
import cv2
import numpy as np
import json
import os
import time
from datetime import datetime
from picamera2 import Picamera2
import math

# -----------------------------------------------------------
# Configuration / Paths
# -----------------------------------------------------------
CALIBRATION_FILE = "camera_calibration.json"
MARKER_DB_FILE = "aruco_marker_positions.json"
OUTPUT_FOLDER = "output_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------------------------------------
# 1) Load Camera Calibration Data
# -----------------------------------------------------------
if not os.path.exists(CALIBRATION_FILE):
    raise FileNotFoundError(f"Missing {CALIBRATION_FILE}")

with open(CALIBRATION_FILE, "r") as f:
    calib_data = json.load(f)

camera_matrix = np.array(calib_data["camera_matrix"], dtype=np.float32)
dist_coeffs   = np.array(calib_data["distortion_coeffs"], dtype=np.float32)

# -----------------------------------------------------------
# 2) Load Marker Database (with known 3D marker corners)
# -----------------------------------------------------------
if not os.path.exists(MARKER_DB_FILE):
    raise FileNotFoundError(f"Missing {MARKER_DB_FILE}")

with open(MARKER_DB_FILE, "r") as f:
    marker_db = json.load(f)

# -----------------------------------------------------------
# 3) Setup ArUco Detector
# -----------------------------------------------------------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

# -----------------------------------------------------------
# 4) Initialize PiCamera2 (outside the function for quick calls)
# -----------------------------------------------------------
picam2 = Picamera2()
# Configure for a reasonable resolution (adjust as needed; 1280x720 here)
config = picam2.create_still_configuration(main={"size": (1280, 720), "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(1)  # Allow camera to warm up

def get_position_from_markers():
    """
    Captures a single frame using the already-initialized PiCamera2,
    detects ArUco markers, computes the camera pose, and returns the camera position (x, y, z)
    and yaw (heading, in degrees). Also saves an annotated image.
    If no markers are detected, returns (None, None).
    """
    # Capture a frame
    frame = picam2.capture_array()
    
    # (Optional) Mirror the frame if desired:
    # frame = cv2.flip(frame, 1)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, _ = detector.detectMarkers(gray)
    
    computed_pose = None
    if ids is not None and len(ids) > 0:
        all_obj_points = []
        all_img_points = []
        for i, marker in enumerate(ids):
            marker_id = str(marker[0])
            if marker_id in marker_db:
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
            ret_pnp, rvec, tvec = cv2.solvePnP(all_obj_points, all_img_points,
                                               camera_matrix, dist_coeffs,
                                               flags=cv2.SOLVEPNP_ITERATIVE)
            if ret_pnp:
                computed_pose = (rvec, tvec)
                # Draw the coordinate axes (trident) at the world origin.
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 50)
    
                # Compute camera position in world coordinates.
                R, _ = cv2.Rodrigues(rvec)
                cam_pos = -R.T.dot(tvec)
    
                # Compute yaw (heading) by projecting the optical axis [0,0,1].
                forward_vector = R.T.dot(np.array([[0], [0], [1]], dtype=np.float32))
                yaw = np.degrees(np.arctan2(forward_vector[1, 0], forward_vector[0, 0]))
                if yaw < 0:
                    yaw += 360
    
                # Overlay camera position and yaw on the frame.
                pos_text = f"Pos: X={cam_pos[0,0]:.1f}, Y={cam_pos[1,0]:.1f}, Z={cam_pos[2,0]:.1f}"
                yaw_text = f"Yaw: {yaw:.1f} deg"
                cv2.putText(frame, pos_text, (20, frame.shape[0] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, yaw_text, (20, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
                print("Camera Position:", cam_pos.flatten())
                print("Camera Yaw:", yaw, "deg")
    
                # Save the annotated image with timestamp.
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_image = os.path.join(OUTPUT_FOLDER, f"{timestamp}_annotated_marker_pose.jpg")
                cv2.imwrite(output_image, frame)
                print(f"Annotated image saved as {output_image}")
    
                return cam_pos.flatten(), yaw
            else:
                print("solvePnP failed.")
                return None, None
        else:
            print("No known markers found for pose estimation.")
            return None, None
    else:
        print("No markers detected.")
        return None, None

def kill_camera():
    picam2.stop()

if __name__ == "__main__":
    try:
        for i in range(10):
            pos, yaw = get_position_from_markers()
            if pos is not None:
                print(f"\n[RESULT] Camera Position: {pos}")
                print(f"[RESULT] Camera Yaw: {yaw:.1f} deg")
            else:
                print("Failed to determine camera pose.")
            time.sleep(1)  # Wait 1 second between captures
    except KeyboardInterrupt:
        print("CTRL+C pressed. Exiting.")
    finally:
        kill_camera()