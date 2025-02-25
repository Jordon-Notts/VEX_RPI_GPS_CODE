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
MARKER_SIZE_MM = 100.0  # Size of the ArUco marker in millimeters
SHOW_STATS_PER_COMPONENT = False

# -----------------------
# CAMERA POSITION IN WORLD (East, North, Up), plus Heading
#  - heading_deg = 0 -> camera faces north (world +Y)
#  - heading_deg = 90 -> camera faces east (world +X)
#  - heading_deg = 180 -> camera faces south, etc.
#  - heading_deg = 270 -> camera faces west
# -----------------------
camera_east        = -400    # East coordinate (mm or other unit)
camera_north       = -400   # North coordinate
camera_up          = 115    # Up (altitude)
camera_heading_deg =   90 # Facing east by default

# Global camera pose in world (to be computed)
camera_global_rvec = None
camera_global_tvec = None

# Collect all detected marker poses for each marker (rvec and tvec)
poses_by_marker = defaultdict(list)
# Also collect the computed marker corners (global 3D) for each pass
marker_corners_by_marker = defaultdict(list)

# -----------------------
# Helper: Build Rotation from "Compass Heading"
# -----------------------
def camera_pose_in_world(east_m, north_m, up_m, heading_deg):
    """
    Return (rvec, tvec) for the camera in the world frame:
      - World coordinate axes: X = east, Y = north, Z = up.
      - heading_deg is measured clockwise from north.
    """
    # Convert heading to radians (negative for CW rotation)
    heading_rad = -np.radians(heading_deg)
    # Rotation about world Z by -heading
    Rz = np.array([
        [ np.cos(heading_rad), -np.sin(heading_rad), 0],
        [ np.sin(heading_rad),  np.cos(heading_rad), 0],
        [          0,                    0,          1]
    ], dtype=np.float32)
    # Rotate about X by -90° so that the camera's local +Z maps to world +Y (when heading=0)
    alpha = -np.pi / 2  # -90 deg
    Rx_neg90 = np.array([
        [1,              0,               0],
        [0,  np.cos(alpha), -np.sin(alpha)],
        [0,  np.sin(alpha),  np.cos(alpha)]
    ], dtype=np.float32)
    Rcw = Rz @ Rx_neg90
    rvec, _ = cv2.Rodrigues(Rcw)
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
# ArUco dictionary & Picamera2 initialization
# -----------------------
aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration(main={"size": (1280,720), "format": "RGB888"}))
picam2.start()

# -----------------------
# JSON I/O for marker positions
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
      - marker_size: marker size (mm or other units)
    Returns a dict of the 4 marker corner points in the WORLD frame.
    The marker model is defined in its local coordinate system:
      top_left:     (-half_size,  half_size, 0)
      top_right:    ( half_size,  half_size, 0)
      bottom_right: ( half_size, -half_size, 0)
      bottom_left:  (-half_size, -half_size, 0)
    """
    half_size = marker_size / 2
    obj_points = {
        "top_left":     [-half_size,  half_size, 0],
        "top_right":    [ half_size,  half_size, 0],
        "bottom_right": [ half_size, -half_size, 0],
        "bottom_left":  [-half_size, -half_size, 0]
    }
    R_cam_global, _    = cv2.Rodrigues(rvec_camera_in_global)
    R_marker_camera, _ = cv2.Rodrigues(rvec_marker_in_camera)
    # Marker rotation in world frame
    R_marker_global = R_cam_global @ R_marker_camera
    # Marker translation in world frame
    t_marker_global = tvec_camera_in_global.reshape(3,1) + R_cam_global @ tvec_marker_in_camera.reshape(3,1)
    corner_points_global = {}
    for name, pt in obj_points.items():
        pt = np.array(pt, dtype=np.float32).reshape(3,1)
        pt_global = R_marker_global @ pt + t_marker_global
        x, y, z = pt_global.flatten()
        corner_points_global[name] = {"x": float(x), "y": float(y), "z": float(z)}
    return corner_points_global

# -----------------------
# Helper: Compute statistics from a list of vectors
# -----------------------
def compute_stats(vec_list):
    """
    Given a list of vectors (each as a (3,1) or (3,) array),
    return a dict with average, std, min, and max for each component.
    """
    arr = np.concatenate([v.reshape(1,3) for v in vec_list], axis=0)
    stats = {
        "avg": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist()
    }
    return stats

# -----------------------
# Helper: Average corners from a list of corner dictionaries
# -----------------------
def average_corners(corner_list):
    """
    Given a list of corner dicts (each with keys "top_left", "top_right", "bottom_right", "bottom_left")
    where each value is a dict with keys "x", "y", "z", return an averaged corner dict.
    """
    avg_corners = {}
    keys = ["top_left", "top_right", "bottom_right", "bottom_left"]
    for key in keys:
        xs = [corners[key]["x"] for corners in corner_list]
        ys = [corners[key]["y"] for corners in corner_list]
        zs = [corners[key]["z"] for corners in corner_list]
        avg_corners[key] = {"x": float(np.mean(xs)),
                            "y": float(np.mean(ys)),
                            "z": float(np.mean(zs))}
    return avg_corners

# -----------------------
# Main function
# -----------------------
def main():
    global camera_global_rvec, camera_global_tvec
    # 1. Compute the camera pose in world coordinates.
    camera_global_rvec, camera_global_tvec = camera_pose_in_world(
        camera_east, camera_north, camera_up, camera_heading_deg
    )
    print("[INFO] Camera extrinsics in world:")
    print("       rvec:", camera_global_rvec.flatten())
    print("       tvec:", camera_global_tvec.flatten())

    # 2. Capture passes and collect poses and marker corners.
    print(f"[INFO] Starting to capture {NUM_PASSES} passes.")
    last_frame = None  # Save the last captured frame for visualization

    for pass_i in range(1, NUM_PASSES+1):
        print(f"\n[INFO] --- PASS {pass_i}/{NUM_PASSES} ---", end="")
        frame = picam2.capture_array()
        last_frame = frame.copy()  # Save a copy for later visualization
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            for i, marker_id_arr in enumerate(ids):
                marker_id = str(marker_id_arr[0])
                marker_corners_img = corners[i][0]  # 4 detected image points

                # 3D model points for the marker (local marker coordinates)
                half_size = MARKER_SIZE_MM / 2
                obj_points = np.array([
                    [-half_size,  half_size, 0],
                    [ half_size,  half_size, 0],
                    [ half_size, -half_size, 0],
                    [-half_size, -half_size, 0]
                ], dtype=np.float32)

                success, rvec_marker_in_camera, tvec_marker_in_camera = cv2.solvePnP(
                    obj_points, marker_corners_img, camera_matrix, dist_coeffs
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

                # Compute the marker's global 3D corners for this pass.
                corners_global = compute_marker_corners_global(
                    camera_global_rvec,
                    camera_global_tvec,
                    rvec_marker_in_camera,
                    tvec_marker_in_camera,
                    MARKER_SIZE_MM
                )
                marker_corners_by_marker[marker_id].append(corners_global)
        else:
            print("[INFO] No markers detected in this pass.")

    # 3. Process each marker: compute statistics and average the corners.
    print("\n[INFO] Finished collecting passes. Computing final marker positions...")
    for marker_id, pose_list in poses_by_marker.items():
        if len(pose_list) == 0:
            print(f"[WARNING] Marker {marker_id} had 0 detections!")
            continue

        # Compute statistics for rvec and tvec.
        rvecs = [pose[0] for pose in pose_list]
        tvecs = [pose[1] for pose in pose_list]
        rvec_stats = compute_stats(rvecs)

        tvec_stats = compute_stats(tvecs)

        # Average pose (if desired, although here we average the corners directly).
        rvec_avg = np.mean(np.concatenate(rvecs, axis=0).reshape(-1, 3), axis=0).reshape(3,1)
        tvec_avg = np.mean(np.concatenate(tvecs, axis=0).reshape(-1, 3), axis=0).reshape(3,1)

        # Instead of computing corners from the averaged pose,
        # average the corners computed on each pass.
        corners_list = marker_corners_by_marker[marker_id]
        avg_corners = average_corners(corners_list)

        # Save marker information to JSON structure.
        marker_positions[marker_id] = {
            "corner_points": avg_corners
            # "rvec_stats": rvec_stats,
            # "tvec_stats": tvec_stats
        }

        # 4. Visualization: Project averaged global corners into the image and overlay text.
        # To project, we need the world-to-camera transform.
        Rcw, _ = cv2.Rodrigues(camera_global_rvec)  # Rotation from camera to world
        Rwc = Rcw.T                                # Inverse: world-to-camera rotation
        t_wc = -Rwc @ camera_global_tvec           # Translation for world-to-camera

        # Create an array of 3D points for the four averaged corners.
        corner_names = ["top_left", "top_right", "bottom_right", "bottom_left"]
        obj_points_3d = []
        for name in corner_names:
            cp = avg_corners[name]
            obj_points_3d.append([cp["x"], cp["y"], cp["z"]])
        obj_points_3d = np.array(obj_points_3d, dtype=np.float32)

        # Convert the world-to-camera rotation matrix to a rotation vector.
        rvec_wc, _ = cv2.Rodrigues(Rwc)
        projected_corners, _ = cv2.projectPoints(obj_points_3d, rvec_wc, t_wc, camera_matrix, dist_coeffs)
        projected_corners = projected_corners.reshape(-1, 2).astype(np.int32)

        vis_img = last_frame.copy()
        # Draw the projected corners and connect them.
        for pt in projected_corners:
            cv2.circle(vis_img, tuple(pt), 5, (0, 255, 0), -1)
        cv2.polylines(vis_img, [projected_corners], isClosed=True, color=(0, 255, 0), thickness=2)

        # Overlay the 3D global coordinates (as multi-line text) at each corner.
        # We'll draw three lines for each corner: x, y, and z.
        line_height = 15  # vertical spacing for the text lines
        for i, name in enumerate(corner_names):
            cp = avg_corners[name]
            pt = tuple(projected_corners[i])
            # Draw x, y, and z values on separate lines.
            cv2.putText(vis_img, f"x={cp['x']:.1f}", (pt[0] + 10, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(vis_img, f"y={cp['y']:.1f}", (pt[0] + 10, pt[1] - 10 + line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(vis_img, f"z={cp['z']:.1f}", (pt[0] + 10, pt[1] - 10 + 2*line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Also overlay text for marker ID and the rvec/tvec statistics.
        text_lines = [f"Marker ID: {marker_id}"]
        for comp, label in zip(range(3), ["rvec_x", "rvec_y", "rvec_z"]):
            avg_val = rvec_stats["avg"][comp]
            std_val = rvec_stats["std"][comp]
            mn_val  = rvec_stats["min"][comp]
            mx_val  = rvec_stats["max"][comp]
            text_lines.append(f"{label}: avg={avg_val:.2f}, std={std_val:.2f}, min={mn_val:.2f}, max={mx_val:.2f}")
        for comp, label in zip(range(3), ["tvec_x", "tvec_y", "tvec_z"]):
            avg_val = tvec_stats["avg"][comp]
            std_val = tvec_stats["std"][comp]
            mn_val  = tvec_stats["min"][comp]
            mx_val  = tvec_stats["max"][comp]
            text_lines.append(f"{label}: avg={avg_val:.2f}, std={std_val:.2f}, min={mn_val:.2f}, max={mx_val:.2f}")

        # Draw the marker stats text on the image.
        x0, y0 = 10, 30
        dy = 25
        for i, line in enumerate(text_lines):
            cv2.putText(vis_img, line, (x0, y0 + i*dy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

        # Save the visualization image.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_filename = os.path.join(SAVE_FOLDER, f"marker_{marker_id}_{timestamp}.png")
        cv2.imwrite(out_filename, vis_img)
        print(f"[INFO] Saved visualization image for marker {marker_id} to {out_filename}")

    # 5. Save final marker positions and stats to JSON.
    save_marker_positions(marker_positions)
    print(f"[INFO] Wrote marker corner positions and stats to {MARKER_LOCATIONS_JSON}")

if __name__ == "__main__":
    main()
