#!/usr/bin/env python3

import os
import json
import cv2
import numpy as np
from datetime import datetime
from picamera2 import Picamera2

# -----------------------------------------------------------
# Configuration / Paths
# -----------------------------------------------------------
CALIBRATION_FILE_JSON = "camera_calibration.json"
MARKER_LOCATIONS_JSON = "aruco_marker_positions.json"
SAVE_FOLDER = "camera_localization_images"  # not used in get_position() but kept from your original
os.makedirs(SAVE_FOLDER, exist_ok=True)

NUM_FRAMES = 10  # Number of frames to average

# -----------------------------------------------------------
# 1) Load Camera Intrinsics
# -----------------------------------------------------------
if not os.path.exists(CALIBRATION_FILE_JSON):
    raise FileNotFoundError(f"Missing {CALIBRATION_FILE_JSON}")

with open(CALIBRATION_FILE_JSON, "r") as f:
    calib_data = json.load(f)

camera_matrix = np.array(calib_data["camera_matrix"], dtype=np.float32)
dist_coeffs   = np.array(calib_data["distortion_coeffs"], dtype=np.float32)

# -----------------------------------------------------------
# 2) Load Known Marker Corners in the World
# -----------------------------------------------------------
if not os.path.exists(MARKER_LOCATIONS_JSON):
    raise FileNotFoundError(f"Missing {MARKER_LOCATIONS_JSON}")

with open(MARKER_LOCATIONS_JSON, "r") as f:
    marker_db = json.load(f)
    
def get_marker_world_corners(marker_id):
    """
    Returns a (4,3) np.array of the known marker corners in the world frame,
    in the order: [top_left, top_right, bottom_right, bottom_left].
    """
    data = marker_db[marker_id]["corner_points"]
    corners_3d = np.array([
        [data["top_left"]["x"],     data["top_left"]["y"],     data["top_left"]["z"]],
        [data["top_right"]["x"],    data["top_right"]["y"],    data["top_right"]["z"]],
        [data["bottom_right"]["x"], data["bottom_right"]["y"], data["bottom_right"]["z"]],
        [data["bottom_left"]["x"],  data["bottom_left"]["y"],  data["bottom_left"]["z"]]
    ], dtype=np.float32)
    return corners_3d

# -----------------------------------------------------------
# 3) ArUco + PiCamera2 Setup
# -----------------------------------------------------------
aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration(main={"size": (1280, 720), "format": "RGB888"}))
picam2.start()

# -----------------------------------------------------------
# 4) Helper Functions
# -----------------------------------------------------------
def average_poses(rvec_tvec_list):
    """
    Naively averages rotation and translation from a list of (rvec, tvec) pairs.
    """
    R_sum = np.zeros((3,3), dtype=np.float64)
    t_sum = np.zeros((3,1), dtype=np.float64)

    for (rvec, tvec) in rvec_tvec_list:
        R_i, _ = cv2.Rodrigues(rvec)
        R_sum += R_i
        t_sum += tvec.reshape(3,1)

    N = len(rvec_tvec_list)
    R_avg = R_sum / N
    t_avg = t_sum / N

    # Re-orthonormalize the rotation matrix via SVD
    U, _, Vt = np.linalg.svd(R_avg)
    R_ortho = U @ Vt

    rvec_out, _ = cv2.Rodrigues(R_ortho)
    return rvec_out.flatten(), t_avg.flatten()

def rotationMatrixToEulerAngles(R):
    """
    Converts a rotation matrix R to Euler angles (roll, pitch, yaw).
    """
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        roll  = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll  = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)

def detect_camera_pose():
    """
    Captures a single image, detects known markers, computes camera->world pose
    for each marker, and returns an average pose if at least one known marker is found.
    Also annotates the image (if needed).
    Returns (rvec_cw, tvec_cw, frame) or (None, None, frame) if no valid detection.
    """
    frame = picam2.capture_array()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners_2d, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        print("[INFO] No markers detected in this frame.")
        return None, None, frame

    camera_poses_cw = []

    for i, marker_id_arr in enumerate(ids):
        marker_id_str = str(marker_id_arr[0])
        if marker_id_str not in marker_db:
            continue

        corners_img_2d = corners_2d[i][0]
        corners_world_3d = get_marker_world_corners(marker_id_str)

        success, rvec_wc, tvec_wc = cv2.solvePnP(
            corners_world_3d,
            corners_img_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            print(f"[ERROR] solvePnP failed for marker ID {marker_id_str}")
            continue

        # Invert to get camera->world pose
        R_wc, _ = cv2.Rodrigues(rvec_wc)
        R_cw = R_wc.T
        t_cw = -R_cw @ tvec_wc

        camera_poses_cw.append((cv2.Rodrigues(R_cw)[0], t_cw))

        # (Optional) Annotation on the image:
        pts = np.int32(corners_img_2d).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec_wc, tvec_wc, 50)
        cxy = corners_img_2d.mean(axis=0).astype(int)
        cv2.putText(frame, f"ID {marker_id_str}", (cxy[0]-10, cxy[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # Also annotate with global marker center
        marker_world_corners = get_marker_world_corners(marker_id_str)
        marker_center = np.mean(marker_world_corners, axis=0)
        global_text = f"Glob: X={marker_center[0]:.1f}, Y={marker_center[1]:.1f}, Z={marker_center[2]:.1f}"
        pt_text = (pts[0][0][0], pts[0][0][1] - 20)
        cv2.putText(frame, global_text, pt_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    if len(camera_poses_cw) == 0:
        print("[INFO] Markers detected, but none matched the DB.")
        return None, None, frame

    if len(camera_poses_cw) > 1:
        rvec_cw, tvec_cw = average_poses(camera_poses_cw)
    else:
        rvec_cw, tvec_cw = camera_poses_cw[0][0], camera_poses_cw[0][1]

    return rvec_cw, tvec_cw, frame

# -----------------------------------------------------------
# 5) get_position Function
# -----------------------------------------------------------
def get_position():
    """
    Captures several frames, averages the camera poses,
    and returns (x, y, yaw) where:
      - x and y are from the averaged translation vector (in the world frame)
      - yaw is the rotation around the Z-axis (in degrees)
    If no valid pose is detected, returns (None, None, None).
    """
    all_poses = []
    
    for i in range(NUM_FRAMES):
        print(f"[INFO] Capturing frame {i+1}/{NUM_FRAMES} ...")
        rvec_cw, tvec_cw, _ = detect_camera_pose()
        if rvec_cw is not None:
            all_poses.append((rvec_cw, tvec_cw))
        else:
            print("[WARN] No valid pose in this frame. Skipping.")

    if len(all_poses) == 0:
        print("[ERROR] Could not find any valid camera poses in the captured frames.")
        return None, None, None

    # Average the poses from all valid frames
    rvec_avg, tvec_avg = average_poses(all_poses)
    R_cw, _ = cv2.Rodrigues(rvec_avg)
    eulers_rad = rotationMatrixToEulerAngles(R_cw)
    eulers_deg = np.degrees(eulers_rad)
    
    # Return x, y (translation) and yaw (rotation about Z)
    x = tvec_avg[0]
    y = tvec_avg[1]
    yaw = eulers_deg[2]
    return x, y, yaw

# -----------------------------------------------------------
# Optional: If run as main, print the position
# -----------------------------------------------------------
if __name__ == "__main__":
    x, y, yaw = get_position()
    if x is not None:
        print("\n[RESULT] Averaged Camera Pose:")
        print(f"  X = {x:.2f}, Y = {y:.2f}, Yaw = {yaw:.1f} degrees")
    else:
        print("Failed to determine camera pose.")
