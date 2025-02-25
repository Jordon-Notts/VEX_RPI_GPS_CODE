#!/usr/bin/env python3

import os
import json
import cv2
import numpy as np
from datetime import datetime
from collections import defaultdict
from picamera2 import Picamera2

# -----------------------
# CONFIGURATIONS
# -----------------------
CALIBRATION_FILE_JSON = "camera_calibration.json"
MARKER_LOCATIONS_JSON = "aruco_marker_positions.json"
SAVE_FOLDER = "marker_mapping_images"
os.makedirs(SAVE_FOLDER, exist_ok=True)

NUM_PASSES = 25
MARKER_SIZE_MM = 100.0  # Size of the ArUco marker (in mm)
SHOW_STATS_PER_COMPONENT = True

# -----------------------
# CAMERA POSITION IN WORLD (East, North, Up), plus Heading
#  - heading_deg = 0 -> camera faces north (world +Y)
#  - heading_deg = 90 -> camera faces east (world +X)
#  - heading_deg = 180 -> camera faces south, etc.
#  - heading_deg = 270 -> camera faces west
# -----------------------
camera_east        = 0.0    # East coordinate
camera_north       = 0.0    # North coordinate
camera_up          = 0.0    # Up (altitude)
camera_heading_deg = 90  # Facing east by default

# Store camera poses (rotation & translation) in these variables
camera_global_rvec = None
camera_global_tvec = None

# Collect all detections so we can average them at the end
poses_by_marker = defaultdict(list)

# -----------------------
# Helper: Build Rotation from "Compass Heading"
# -----------------------
def camera_pose_in_world(east_m, north_m, up_m, heading_deg):
    """
    Return (rvec, tvec) for the camera in the world frame:
      - X = east, Y = north, Z = up (right-handed)
      - heading_deg measured clockwise from north:
        heading=0 means local +Z -> world +Y
        heading=90 means local +Z -> world +X
    """
    # Convert heading to radians, noting that standard math rotation about +Z is CCW,
    # but heading is typically CW from north.
    heading_rad = -np.radians(heading_deg)

    # Rotation about world Z by -heading
    Rz = np.array([
        [ np.cos(heading_rad), -np.sin(heading_rad), 0],
        [ np.sin(heading_rad),  np.cos(heading_rad), 0],
        [          0,                    0,          1]
    ], dtype=np.float32)

    # Then rotate about X by -90° to send camera local +Z -> world +Y (if heading=0)
    alpha = -np.pi / 2  # -90 deg
    Rx_neg90 = np.array([
        [1,              0,               0],
        [0,  np.cos(alpha), -np.sin(alpha)],
        [0,  np.sin(alpha),  np.cos(alpha)]
    ], dtype=np.float32)

    # Final rotation camera->world
    Rcw = Rz @ Rx_neg90

    # Convert rotation matrix to Rodrigues vector
    rvec, _ = cv2.Rodrigues(Rcw)

    # The translation is just the camera's world position
    tvec = np.array([east_m, north_m, up_m], dtype=np.float32)

    return rvec, tvec


# -----------------------
# Load camera intrinsics
# -----------------------
if not os.path.exists(CALIBRATION_FILE_JSON):
    raise FileNotFoundError(f"Missing {CALIBRATION_FILE_JSON}; please provide camera calibration.")

with open(CALIBRATION_FILE_JSON, "r") as f:
    calib_data = json.load(f)

camera_matrix = np.array(calib_data["camera_matrix"])
dist_coeffs   = np.array(calib_data["distortion_coeffs"])

# -----------------------
# ArUco dictionary & Picamera2
# -----------------------
aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration(main={"size": (1280, 720)}))
picam2.start()

# -----------------------
# JSON I/O
# -----------------------
def load_marker_positions():
    if not os.path.exists(MARKER_LOCATIONS_JSON):
        with open(MARKER_LOCATIONS_JSON, "w") as f:
            json.dump({}, f)
        return {}
    with open(MARKER_LOCATIONS_JSON, "r") as f:
        return json.load(f)

def save_marker_positions(data):
    with open(MARKER_LOCATIONS_JSON, "w") as f:
        json.dump(data, f, indent=2)

marker_positions = load_marker_positions()

# -----------------------
# Compute Marker Corners in Global Frame
# -----------------------
def compute_marker_corners_global(rvec_camera_in_global, tvec_camera_in_global,
                                  rvec_marker_in_camera, tvec_marker_in_camera,
                                  marker_size):
    """
    Given:
      - rvec_camera_in_global, tvec_camera_in_global: camera pose in WORLD
      - rvec_marker_in_camera, tvec_marker_in_camera: marker pose in CAMERA
      - marker_size: size of marker in same units (mm or m)
    Returns a dict of the 4 marker corner points in the WORLD frame.
    """
    half_size = marker_size / 2
    obj_points = {
        "top_left":     [-half_size,  half_size, 0],
        "top_right":    [ half_size,  half_size, 0],
        "bottom_right": [ half_size, -half_size, 0],
        "bottom_left":  [-half_size, -half_size, 0]
    }

    # Convert rotation vectors to rotation matrices
    R_cam_global, _    = cv2.Rodrigues(rvec_camera_in_global)
    R_marker_camera, _ = cv2.Rodrigues(rvec_marker_in_camera)

    # Marker rotation in world = R_cam_global * R_marker_camera
    R_marker_global = R_cam_global @ R_marker_camera

    # Marker translation in world
    t_marker_global = (tvec_camera_in_global.reshape(3,1)
                       + R_cam_global @ tvec_marker_in_camera.reshape(3,1))

    # Compute global corner points
    corner_points_global = {}
    for name, obj_pt in obj_points.items():
        obj_pt = np.array(obj_pt, dtype=np.float32).reshape(3,1)
        pt_global = R_marker_global @ obj_pt + t_marker_global
        x, y, z = pt_global.flatten()
        corner_points_global[name] = {
            "x": float(x),
            "y": float(y),
            "z": float(z)
        }

    return corner_points_global

def main():
    # ----------------------------------------------------------
    # 1. Compute camera pose in world from (east, north, up, heading)
    # ----------------------------------------------------------
    global camera_global_rvec, camera_global_tvec
    camera_global_rvec, camera_global_tvec = camera_pose_in_world(
        camera_east,
        camera_north,
        camera_up,
        camera_heading_deg
    )
    
    print("[INFO] Camera extrinsics in world:")
    print("       rvec:", camera_global_rvec.flatten())
    print("       tvec:", camera_global_tvec.flatten())

    # ----------------------------------------------------------
    # 2. Capture passes and collect poses for each marker
    # ----------------------------------------------------------
    print(f"[INFO] Starting to capture {NUM_PASSES} passes.")

    for pass_i in range(1, NUM_PASSES+1):
        print(f"\n[INFO] --- PASS {pass_i}/{NUM_PASSES} ---")

        frame = picam2.capture_array()
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            for i, marker_id_arr in enumerate(ids):
                marker_id = str(marker_id_arr[0])
                marker_corners = corners[i][0]

                # 3D model points of the marker (local to marker center)
                half_size = MARKER_SIZE_MM / 2
                obj_points = np.array([
                    [-half_size,  half_size, 0],
                    [ half_size,  half_size, 0],
                    [ half_size, -half_size, 0],
                    [-half_size, -half_size, 0]
                ], dtype=np.float32)

                success, rvec_marker_in_camera, tvec_marker_in_camera = cv2.solvePnP(
                    obj_points, marker_corners, camera_matrix, dist_coeffs
                )
                if not success:
                    print(f"[ERROR] solvePnP failed for marker {marker_id}")
                    continue

                rx, ry, rz = rvec_marker_in_camera.flatten()
                tx, ty, tz = tvec_marker_in_camera.flatten()
                print(f"    Marker {marker_id} pass {pass_i}: "
                      f"r=({rx:.2f},{ry:.2f},{rz:.2f}), "
                      f"t=({tx:.2f},{ty:.2f},{tz:.2f})")

                poses_by_marker[marker_id].append((rvec_marker_in_camera, tvec_marker_in_camera))
        else:
            print("[INFO] No markers detected in this pass.")

    # ----------------------------------------------------------
    # 3. Average poses and compute global corners
    # ----------------------------------------------------------
    print("\n[INFO] Finished collecting passes. Computing final marker positions...")

    for marker_id, pose_list in poses_by_marker.items():
        if len(pose_list) == 0:
            print(f"[WARNING] Marker {marker_id} had 0 detections!")
            continue

        # Average rvec, tvec across all passes
        rvec_avg = np.mean([r for r, _ in pose_list], axis=0)
        tvec_avg = np.mean([t for _, t in pose_list], axis=0)

        corner_points_global = compute_marker_corners_global(
            camera_global_rvec,
            camera_global_tvec,
            rvec_avg,
            tvec_avg,
            MARKER_SIZE_MM
        )

        marker_positions[marker_id] = {
            "corner_points": corner_points_global
        }

    # ----------------------------------------------------------
    # 4. Save to JSON
    # ----------------------------------------------------------
    save_marker_positions(marker_positions)
    print(f"[INFO] Wrote marker corner positions to {MARKER_LOCATIONS_JSON}")


if __name__ == "__main__":
    main()