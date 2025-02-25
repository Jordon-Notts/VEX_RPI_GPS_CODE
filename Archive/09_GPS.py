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

camera_matrix = np.array(calib_data["camera_matrix"])  # e.g. shape (3,3)
dist_coeffs   = np.array(calib_data["distortion_coeffs"])  # e.g. shape (5,)

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
        R_i, _ = cv2.Rodrigues(rvec)
        R_sum += R_i
        t_sum += tvec.reshape(3,1)

    N = len(rvec_tvec_list)
    R_avg = R_sum / N
    t_avg = t_sum / N

    # Orthonormalize R_avg
    U, s, Vt = np.linalg.svd(R_avg)
    R_ortho = U @ Vt

    rvec_final, _ = cv2.Rodrigues(R_ortho)
    return rvec_final.flatten(), t_avg.flatten()

# -----------------------------------------------------------
# 7) Main: Detect ArUco, Solve for Camera Pose
# -----------------------------------------------------------
def locate_camera():

    frame = picam2.capture_array()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners_2d, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        print("[WARNING] No markers detected.")
        return None, frame

    candidate_poses = []

    for i, marker_id_arr in enumerate(ids):
        marker_id_str = str(marker_id_arr[0])
        if marker_id_str not in marker_db:
            continue

        # 1) 2D corners from detection (4x2)
        corners_img_2d = corners_2d[i][0]

        # 2) 3D corners from JSON (4x3) in WORLD coords
        corners_world_3d = get_marker_global_corners(marker_db[marker_id_str])

        # 3) solvePnP => (rvec_wc, tvec_wc) transforms a world point into camera coords
        
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

        # --- Convert rvec_wc to a 3x3 rotation matrix ---
        R_wc, _ = cv2.Rodrigues(rvec_wc)  # world->camera

        # --- Invert world->camera => camera->world ---
        R_cw = R_wc.T
        t_cw = -R_cw @ tvec_wc  # camera->world translation

        # We'll store (rvec_cw, t_cw) in candidate_poses
        rvec_cw, _ = cv2.Rodrigues(R_cw)
        candidate_poses.append((rvec_cw, t_cw))

        # For debugging: draw 3D axes using (rvec_wc, tvec_wc) on the marker
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                          rvec_wc, tvec_wc, 50)

        # Label the marker on the image
        cxy = corners_img_2d.mean(axis=0).astype(int)
        cv2.putText(frame, f"ID {marker_id_str}", (cxy[0]-20, cxy[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    if not candidate_poses:
        print("[WARNING] Markers detected, but none found in DB.")
        return None, frame

    # 4) If multiple markers recognized, average the resulting camera->world poses
    rvec_cam_world, tvec_cam_world = average_poses(candidate_poses)

    # 5) (Optional) If your camera is offset from the robot center, apply that offset:
    #    Suppose the camera is +0.15m forward, +0.05m left, +0.10m up of the robot center,
    #    in the camera’s local coordinate system. Then the robot's origin is:
    #        P_robot_in_camera = [-0.15, -0.05, -0.10], etc.
    #    Or more commonly the camera is "in front" => [+0.15, 0, +0.10], depends on sign convention.

    # Example offset: "camera is 0.10m forward, 0.02m left, 0.0m up from the robot center"
    # (Change signs & values to match your real geometry)
    offset_camera_in_robot = np.array([0.10, 0.02, 0.0], dtype=np.float32)

    # We want "robot center" in the world. So if X_cam is a point in camera coords,
    # then X_world = R_cw * X_cam + t_cw.
    R_cw_final, _ = cv2.Rodrigues(rvec_cam_world)
    tvec_robot_world = tvec_cam_world + R_cw_final @ (-offset_camera_in_robot)

    # We also can define rvec_robot_world = same rotation as the camera if the robot
    # is not rotating relative to the camera. Or if the robot's "facing direction"
    # is aligned with camera +Z axis, you keep the same orientation. Or maybe the
    # robot "facing direction" aligns with camera +X. That depends on your geometry.
    rvec_robot_world = rvec_cam_world  # For a simple case: the robot faces where the camera faces

    # 6) (Optional) compute "yaw" if your global is X=East, Y=North, Z=Up
    # The camera's "forward" is local +Z => 3rd column of R_cw
    R_cam_world, _ = cv2.Rodrigues(rvec_cam_world)
    forward_vec = R_cam_world[:,2]  # Z-axis in global
    yaw_rad = np.arctan2(forward_vec[1], forward_vec[0])  # Y over X
    yaw_deg = (np.degrees(yaw_rad) + 360) % 360

    print(f"[INFO] Robot pose => X={tvec_robot_world[0]:.1f}, "
          f"Y={tvec_robot_world[1]:.1f}, Z={tvec_robot_world[2]:.1f}, "
          f"Yaw={yaw_deg:.1f}°")

    # Annotate the final info
    cv2.putText(frame, f"Robot: X={tvec_robot_world[0]:.2f}, "
                       f"Y={tvec_robot_world[1]:.2f}, Z={tvec_robot_world[2]:.2f}",
                (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    cv2.putText(frame, f"Yaw={yaw_deg:.1f} deg", (20,70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    return (rvec_robot_world, tvec_robot_world, yaw_deg), frame

# -----------------------------------------------------------
# 8) Run & Save Annotated Image
# -----------------------------------------------------------
if __name__ == "__main__":
    result, annotated_frame = locate_camera()

    # Save annotated
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(SAVE_FOLDER, f"camera_pose_{ts}.jpg")
    cv2.imwrite(out_path, annotated_frame)
    print(f"[INFO] Saved annotated image => {out_path}")

    if result:
        rvec_final, tvec_final, yaw_deg = result
        print(f"[RESULT] Camera pose => X={tvec_final[0]:.1f}, "
              f"Y={tvec_final[1]:.1f}, Z={tvec_final[2]:.1f}, "
              f"Yaw={yaw_deg:.1f}°")
    else:
        print("[INFO] No camera pose could be computed.")
