#!/usr/bin/env python3

import os
import json
import cv2
import numpy as np
from datetime import datetime
from picamera2 import Picamera2

# -----------------------------------------------------------
# 1) Configuration / Paths
# -----------------------------------------------------------
CALIBRATION_FILE_JSON = "camera_calibration.json"
MARKER_LOCATIONS_JSON = "aruco_marker_positions.json"
SAVE_FOLDER = "camera_localization_images"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# -----------------------------------------------------------
# 2) Load Camera Intrinsics
# -----------------------------------------------------------
if not os.path.exists(CALIBRATION_FILE_JSON):
    raise FileNotFoundError(f"Missing {CALIBRATION_FILE_JSON}; please provide camera calibration.")

with open(CALIBRATION_FILE_JSON, "r") as f:
    calib_data = json.load(f)

camera_matrix = np.array(calib_data["camera_matrix"])  # shape (3,3)
dist_coeffs   = np.array(calib_data["distortion_coeffs"])  # shape (k,)

# -----------------------------------------------------------
# 3) Load Marker Corners (global coords)
# -----------------------------------------------------------
if not os.path.exists(MARKER_LOCATIONS_JSON):
    raise FileNotFoundError(f"Missing {MARKER_LOCATIONS_JSON}; please store marker global info.")

with open(MARKER_LOCATIONS_JSON, "r") as f:
    marker_db = json.load(f)
# marker_db[marker_id]["corner_points"]["top_left"] => {x, y, z}, etc.

def get_marker_global_corners(marker_info):
    """
    Extract the 4 corner points in global coords (x,y,z) from the JSON.
    Must preserve the same order that ArUco detectMarkers uses:
      [top_left, top_right, bottom_right, bottom_left].
    Returns a (4,3) float32 array.
    """
    cp = marker_info["corner_points"]
    corners_3d = np.array([
        [cp["top_left"]["x"],     cp["top_left"]["y"],     cp["top_left"]["z"]],
        [cp["top_right"]["x"],    cp["top_right"]["y"],    cp["top_right"]["z"]],
        [cp["bottom_right"]["x"], cp["bottom_right"]["y"], cp["bottom_right"]["z"]],
        [cp["bottom_left"]["x"],  cp["bottom_left"]["y"],  cp["bottom_left"]["z"]]
    ], dtype=np.float32)
    return corners_3d

# -----------------------------------------------------------
# 4) ArUco + PiCamera2
# -----------------------------------------------------------
aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration(main={"size": (1280, 720)}))
picam2.start()

# -----------------------------------------------------------
# 5) Helper: Convert Rotation Matrix -> Euler Angles
#    (Assuming a particular convention, e.g., ZYX)
# -----------------------------------------------------------
def rotationMatrixToEulerAngles(R):
    """
    Convert a 3x3 rotation matrix R into Euler angles [roll, pitch, yaw].
    Using a ZYX convention (yaw around Z, pitch around Y, roll around X).
    Returns angles in radians.
    """
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        roll  = np.arctan2(R[2, 1], R[2, 2])  # X-axis rotation
        pitch = np.arctan2(-R[2, 0], sy)      # Y-axis rotation
        yaw   = np.arctan2(R[1, 0], R[0, 0])  # Z-axis rotation
    else:
        # Fallback for near gimbal lock
        roll  = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = 0.0

    return np.array([roll, pitch, yaw], dtype=np.float64)

# -----------------------------------------------------------
# 6) Average Poses (if multiple markers)
# -----------------------------------------------------------
def average_poses(rvec_tvec_list):
    """
    Naive averaging of rotation + translation:
      1) Convert each rvec->matrix, sum them up
      2) Re-orthonormalize via SVD
      3) Average translations
    """
    R_sum = np.zeros((3,3), dtype=np.float64)
    t_sum = np.zeros((3,1), dtype=np.float64)

    for (rvec, tvec) in rvec_tvec_list:
        R_i, _ = cv2.Rodrigues(rvec)  # Convert axis-angle to 3x3 matrix
        R_sum += R_i
        t_sum += tvec.reshape(3,1)

    N = len(rvec_tvec_list)
    R_avg = R_sum / N
    t_avg = t_sum / N

    # Orthonormalize R_avg via SVD
    U, _, Vt = np.linalg.svd(R_avg)
    R_ortho = U @ Vt

    rvec_final, _ = cv2.Rodrigues(R_ortho)
    return rvec_final.flatten(), t_avg.flatten()

# -----------------------------------------------------------
# 7) Main: Detect ArUco, Solve for Camera Pose
# -----------------------------------------------------------
def locate_camera():
    """
    1) Capture an image with Picamera2.
    2) Detect ArUco markers.
    3) For each marker in our DB, run solvePnP to get world->camera.
    4) Invert to get camera->world, store in candidate_poses.
    5) If multiple markers are found, average the camera->world pose.
    6) (Optional) apply camera offset if the camera is not at the robot origin.
    7) Return final camera position/orientation in the global frame.
    """

    # --- 1) Capture image, convert to grayscale ---
    frame = picam2.capture_array()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- 2) Detect ArUco markers ---
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners_2d, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        print("[WARNING] No markers detected.")
        return None, frame

    candidate_poses = []

    for i, marker_id_arr in enumerate(ids):
        marker_id_str = str(marker_id_arr[0])
        if marker_id_str not in marker_db:
            # Marker not in our database, skip it
            continue

        # (4x2) corners in the image
        corners_img_2d = corners_2d[i][0]

        # (4x3) corners in global/world
        corners_world_3d = get_marker_global_corners(marker_db[marker_id_str])

        # --- 3) solvePnP => world->camera ---
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

        # --- Convert rvec_wc -> rotation matrix (world->camera) ---
        R_wc, _ = cv2.Rodrigues(rvec_wc)

        # --- 4) Invert to get camera->world ---
        R_cw = R_wc.T
        t_cw = -R_cw @ tvec_wc  # camera->world translation

        # We'll store (rvec_cw, t_cw) for each marker
        rvec_cw, _ = cv2.Rodrigues(R_cw)
        candidate_poses.append((rvec_cw, t_cw))

        # For visualization: draw the 3D axes on the marker
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                          rvec_wc, tvec_wc, 50)

        # Label the marker ID on the image
        cxy = corners_img_2d.mean(axis=0).astype(int)
        cv2.putText(frame, f"ID {marker_id_str}", (cxy[0]-20, cxy[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    if not candidate_poses:
        print("[WARNING] Markers detected, but none found in DB.")
        return None, frame

    # --- If multiple markers recognized, average the camera->world poses ---
    rvec_cam_world, tvec_cam_world = average_poses(candidate_poses)

    # --- 5) Optional camera offset if camera is not at the robot's origin ---
    # For example: camera is +0.10m forward (x), +0.02m to the left (y), 0.0m up (z)
    # in camera's own coordinate system:
    offset_camera_in_robot = np.array([+0.10, +0.02, 0.0], dtype=np.float32)
    # We want the robot origin in the world => T_robot_world:
    R_cw_final, _ = cv2.Rodrigues(rvec_cam_world)
    t_robot_world = tvec_cam_world + R_cw_final @ (-offset_camera_in_robot)

    # If your robot's heading is the same as the camera's heading, keep the same rvec.
    # Otherwise, adjust based on how the camera is mounted.
    rvec_robot_world = rvec_cam_world

    # --- 6) Convert to Euler angles in degrees (camera->world or robot->world) ---
    # We'll do it for the final orientation (robot->world).
    R_robot_world, _ = cv2.Rodrigues(rvec_robot_world)
    euler_rad = rotationMatrixToEulerAngles(R_robot_world)  # roll, pitch, yaw in radians
    euler_deg = np.degrees(euler_rad)                      # convert to degrees

    # For a typical XY-plane environment, you might care mostly about yaw:
    # the local +Z axis is forward for the camera, so let's define yaw as:
    forward_vec = R_robot_world[:,2]  # local Z in global
    yaw_rad = np.arctan2(forward_vec[1], forward_vec[0])
    yaw_deg = (np.degrees(yaw_rad) + 360) % 360

    # --- Print out the final result ---
    print("[INFO] Camera/Robot Pose in Global Coordinates")
    print(f"   X = {t_robot_world[0]:.3f} m")
    print(f"   Y = {t_robot_world[1]:.3f} m")
    print(f"   Z = {t_robot_world[2]:.3f} m")
    print(f"   Roll/Pitch/Yaw (deg) = {euler_deg[0]:.1f}, "
          f"{euler_deg[1]:.1f}, {euler_deg[2]:.1f}")
    print(f"   (Alternative Yaw calc) = {yaw_deg:.1f} deg")

    # --- Annotate the frame with numeric results ---
    cv2.putText(frame, f"X={t_robot_world[0]:.2f}, Y={t_robot_world[1]:.2f}, Z={t_robot_world[2]:.2f}",
                (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    cv2.putText(frame, f"Yaw={yaw_deg:.1f} deg", (20,70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    # Return whichever final data you want
    return {
        "rvec_robot_world": rvec_robot_world,
        "tvec_robot_world": t_robot_world,
        "euler_deg": euler_deg,
        "yaw_deg": yaw_deg
    }, frame

# -----------------------------------------------------------
# 8) Run & Save Annotated Image
# -----------------------------------------------------------
if __name__ == "__main__":
    result, annotated_frame = locate_camera()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(SAVE_FOLDER, f"camera_pose_{ts}.jpg")
    cv2.imwrite(out_path, annotated_frame)
    print(f"[INFO] Saved annotated image => {out_path}")

    if result:
        print("[RESULT] Final Pose:")
        print(f"  X = {result['tvec_robot_world'][0]:.2f}, "
              f"Y = {result['tvec_robot_world'][1]:.2f}, "
              f"Z = {result['tvec_robot_world'][2]:.2f}")
        print(f"  Euler angles (deg) = {result['euler_deg']}")
        print(f"  Yaw = {result['yaw_deg']:.1f} deg")
    else:
        print("[INFO] No camera pose could be computed.")
