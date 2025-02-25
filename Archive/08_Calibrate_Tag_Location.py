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

NUM_PASSES = 10
MARKER_SIZE_MM = 100.0  # If all markers are the same size
SHOW_STATS_PER_COMPONENT = True  # If True, prints mean, std, min, max for each of r_x, r_y, r_z, t_x, t_y, t_z

# Camera's known global pose
camera_global_x = 0.0
camera_global_y = 0.0
camera_global_z = 0.0
camera_global_yaw_deg = 180  # e.g., 0 => camera forward is global +Z

def rvec_from_yaw(yaw_degrees):
    """Rodrigues vector for rotation about Z by 'yaw_degrees'."""
    yaw_rad = np.radians(yaw_degrees)
    return np.array([-1.5708, 0.0, yaw_rad], dtype=np.float32)

camera_global_rvec = rvec_from_yaw(camera_global_yaw_deg)
camera_global_tvec = np.array([camera_global_x, camera_global_y, camera_global_z], dtype=np.float32)

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
# Utility: average rotation
# -----------------------
def average_pose(pose_list):
    """
    Averages a list of (rvec, tvec) by:
      - converting each rvec -> R_i (3x3),
      - summing them,
      - normalizing via SVD,
      - averaging tvec normally.
    Returns (rvec_avg, tvec_avg).
    """
    R_sum = np.zeros((3,3), dtype=np.float64)
    t_sum = np.zeros((3,1), dtype=np.float64)

    for (rvec, tvec) in pose_list:
        rvec = rvec.reshape(3,)
        tvec = tvec.reshape(3,1)
        R_i, _ = cv2.Rodrigues(rvec)
        R_sum += R_i
        t_sum += tvec

    N = len(pose_list)
    R_avg = R_sum / N
    t_avg = t_sum / N

    # Orthonormalize R_avg
    U, s, Vt = np.linalg.svd(R_avg)
    R_ortho = U @ Vt
    rvec_avg, _ = cv2.Rodrigues(R_ortho)
    return rvec_avg.flatten(), t_avg.flatten()

def compute_marker_global_pose(rvec_camera_in_global,
                               tvec_camera_in_global,
                               rvec_marker_in_camera,
                               tvec_marker_in_camera):
    """
    marker_in_global = camera_in_global * marker_in_camera
    """
    R_cam_global, _ = cv2.Rodrigues(rvec_camera_in_global)
    R_marker_camera, _ = cv2.Rodrigues(rvec_marker_in_camera)

    R_marker_global = R_cam_global @ R_marker_camera
    rvec_marker_in_global, _ = cv2.Rodrigues(R_marker_global)

    t_marker_in_global = tvec_camera_in_global.reshape(3,1) + R_cam_global @ tvec_marker_in_camera.reshape(3,1)
    return rvec_marker_in_global.flatten(), t_marker_in_global.flatten()

# -----------------------
# Data structures
# -----------------------
# For each marker, store a list of (rvec, tvec) from each pass
poses_by_marker = defaultdict(list)

def main():
    print(f"[INFO] Starting to capture {NUM_PASSES} passes (no images saved during passes).")

    # 1) Collect N passes of data
    for pass_i in range(1, NUM_PASSES+1):
        print(f"\n[INFO] --- PASS {pass_i}/{NUM_PASSES} ---")

        # Capture a frame
        frame = picam2.capture_array()
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            for i, marker_id_arr in enumerate(ids):
                marker_id = str(marker_id_arr[0])  # '0','1','2'...
                marker_corners = corners[i][0]

                # Solve PnP
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

                # For debugging, print each pass's local (rvec, tvec)
                rx, ry, rz = rvec_marker_in_camera.flatten()
                tx, ty, tz = tvec_marker_in_camera.flatten()
                print(f"    Marker {marker_id} pass {pass_i}: "
                      f"r=({rx:.2f},{ry:.2f},{rz:.2f}), "
                      f"t=({tx:.2f},{ty:.2f},{tz:.2f})")

                # Store it
                poses_by_marker[marker_id].append((rvec_marker_in_camera, tvec_marker_in_camera))
        else:
            print("[INFO] No markers detected this pass.")

    print("\n[INFO] Finished collecting passes. Computing final stats & averaging...")

    # 2) For each marker, gather stats and compute average
    #    We'll store final results in 'marker_positions' (the JSON dict)
    for marker_id, pose_list in poses_by_marker.items():
        if len(pose_list) == 0:
            print(f"[WARNING] Marker {marker_id} had 0 detections total!")
            continue

        # Build arrays for stats
        # We'll store r_x, r_y, r_z, t_x, t_y, t_z separately
        # so we can compute mean, std, min, max easily
        # Then we'll do the rotation averaging for the final pose.
        data_array = np.zeros((len(pose_list), 6), dtype=np.float64)
        for i, (rvec, tvec) in enumerate(pose_list):
            rx, ry, rz = rvec.flatten()
            tx, ty, tz = tvec.flatten()
            data_array[i] = [rx, ry, rz, tx, ty, tz]

        # Basic stats for each component
        mean_vals = data_array.mean(axis=0)  # shape (6,)
        std_vals  = data_array.std(axis=0)   # shape (6,)
        min_vals  = data_array.min(axis=0)   # shape (6,)
        max_vals  = data_array.max(axis=0)   # shape (6,)

        # Print to console
        print(f"\n[INFO] Marker {marker_id} Stats (Local Pose in Camera Frame):")
        components = ["r_x", "r_y", "r_z", "t_x", "t_y", "t_z"]
        for c_idx, c_name in enumerate(components):
            print(f"   {c_name:4s}: "
                  f"mean={mean_vals[c_idx]:.3f}, "
                  f"std={std_vals[c_idx]:.3f}, "
                  f"min={min_vals[c_idx]:.3f}, "
                  f"max={max_vals[c_idx]:.3f}")

        # Now do rotation averaging for final pose
        rvec_avg_cam, tvec_avg_cam = average_pose(pose_list)

        # Convert that average local pose to marker_in_global
        rvec_marker_in_global, tvec_marker_in_global = compute_marker_global_pose(
            camera_global_rvec,
            camera_global_tvec,
            rvec_avg_cam,
            tvec_avg_cam
        )
        ax, ay, az = rvec_marker_in_global
        x, y, z    = tvec_marker_in_global

        # Overwrite in the JSON structure
        marker_positions[marker_id] = {
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "ax": float(ax),
            "ay": float(ay),
            "az": float(az),
            "marker_size_mm": MARKER_SIZE_MM
        }

    # 3) Save JSON
    save_marker_positions(marker_positions)
    print(f"[INFO] Wrote final marker poses to {MARKER_LOCATIONS_JSON}")

    # 4) Capture one final image, re-detect markers, and annotate with stats
    #    (If you only want to show the final averaged pose, you can do that, but
    #     we'll just detect again and label each marker.)
    print("[INFO] Capturing final image for annotation...")
    final_frame = picam2.capture_array()
    gray_final  = cv2.cvtColor(final_frame, cv2.COLOR_BGR2GRAY)

    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, _ = detector.detectMarkers(gray_final)

    if ids is not None and len(ids) > 0:
        for i, marker_id_arr in enumerate(ids):
            marker_id = str(marker_id_arr[0])
            if marker_id not in poses_by_marker:
                # We didn't record passes for this marker, or it wasn't detected earlier
                continue

            # We'll overlay text near the marker
            center_pt = corners[i][0].mean(axis=0).astype(int)
            # For each marker, show stats and pass count
            pose_list = poses_by_marker[marker_id]
            pass_count = len(pose_list)
            # We'll also re-use the stats we computed above by re-building (or
            # store them in a dictionary if you'd like).
            data_array = np.zeros((pass_count, 6), dtype=np.float64)
            for j, (rvec, tvec) in enumerate(pose_list):
                rx, ry, rz = rvec.flatten()
                tx, ty, tz = tvec.flatten()
                data_array[j] = [rx, ry, rz, tx, ty, tz]

            mean_vals = data_array.mean(axis=0)
            std_vals  = data_array.std(axis=0)
            min_vals  = data_array.min(axis=0)
            max_vals  = data_array.max(axis=0)

            # Prepare multi-line text
            # Example: Marker ID, passes, then for each component: mean, std, min, max
            text_lines = [
                f"Marker ID {marker_id}",
                f"Passes: {pass_count}"
            ]
            if SHOW_STATS_PER_COMPONENT:
                comps = ["rX", "rY", "rZ", "tX", "tY", "tZ"]
                for c_idx, c_name in enumerate(comps):
                    text_lines.append(
                        f"{c_name}: mean={mean_vals[c_idx]:.2f}, "
                        f"std={std_vals[c_idx]:.2f}, "
                        f"min={min_vals[c_idx]:.2f}, "
                        f"max={max_vals[c_idx]:.2f}"
                    )

            # Put the text on the final image
            x0, y0 = center_pt[0], center_pt[1]
            offset_y = 0
            for line in text_lines:
                cv2.putText(final_frame, line, (x0-100, y0 + offset_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                offset_y += 20

        # Optionally draw the axes again
        # If you want to re-run solvePnP for the final image, you can do so, or skip.

    # Save the single final annotated image
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_out_path = os.path.join(SAVE_FOLDER, f"final_annotated_{ts}.jpg")
    cv2.imwrite(final_out_path, final_frame)
    print(f"[INFO] Final annotated image saved => {final_out_path}")

if __name__ == "__main__":
    main()
