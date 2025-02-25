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

def get_marker_global_corners(marker_info):
    """
    Extract the 4 corner points in global coords (x,y,z) from the JSON.
    Must preserve the order ArUco uses: [top_left, top_right, bottom_right, bottom_left].
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
# 5) Averaging Utility
# -----------------------------------------------------------
def average_poses(rvec_tvec_list):
    """
    Naive averaging of rotation + translation:
      1) Convert each rvec->matrix, sum them up
      2) Re-orthonormalize via SVD
      3) Average translations
    Returns (rvec_final, tvec_final)
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
    U, _, Vt = np.linalg.svd(R_avg)
    R_ortho = U @ Vt

    rvec_final, _ = cv2.Rodrigues(R_ortho)
    return rvec_final.flatten(), t_avg.flatten()

# -----------------------------------------------------------
# 6) Detect + Solve Pose for a Single Frame
#    Returns a single average camera->world pose if multiple markers
# -----------------------------------------------------------
def detect_camera_pose():
    """
    Capture one frame and estimate the camera->world pose by
    inverting each marker's world->camera pose from solvePnP.
    If multiple markers are detected, average them.
    Returns (rvec_cw, tvec_cw), annotated_frame or (None, frame) if fail.
    """

    frame = picam2.capture_array()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners_2d, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        print("[WARNING] No markers detected in this frame.")
        return None, frame

    candidate_poses = []

    for i, marker_id_arr in enumerate(ids):
        marker_id_str = str(marker_id_arr[0])
        if marker_id_str not in marker_db:
            # Skip markers not in our DB
            continue

        # 2D corners from detection (4x2)
        corners_img_2d = corners_2d[i][0]

        # 3D corners from JSON (4x3) in WORLD coords
        corners_world_3d = get_marker_global_corners(marker_db[marker_id_str])

        # solvePnP => (rvec_wc, tvec_wc) transforms a world point into camera coords
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

        # World->camera rotation matrix
        R_wc, _ = cv2.Rodrigues(rvec_wc)
        # Invert to get camera->world
        R_cw = R_wc.T
        t_cw = -R_cw @ tvec_wc

        # Store as (rvec_cw, t_cw)
        rvec_cw, _ = cv2.Rodrigues(R_cw)
        candidate_poses.append((rvec_cw, t_cw))

        # Optional: draw 3D axes on the marker
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec_wc, tvec_wc, 50)

        # Label the marker ID in the image
        cxy = corners_img_2d.mean(axis=0).astype(int)
        cv2.putText(frame, f"ID {marker_id_str}", (cxy[0]-20, cxy[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    if not candidate_poses:
        print("[WARNING] Markers detected, but none found in DB.")
        return None, frame

    # If multiple markers recognized in this frame, average them
    rvec_final, tvec_final = average_poses(candidate_poses)

    return (rvec_final, tvec_final), frame

# -----------------------------------------------------------
# 7) Main: Capture 10 frames, average camera pose
# -----------------------------------------------------------
if __name__ == "__main__":
    all_poses = []
    annotated_frame = None

    # Capture and accumulate 10 poses
    for i in range(10):
        pose, frame = detect_camera_pose()
        if pose is not None:
            all_poses.append(pose)
            annotated_frame = frame  # store last frame for annotation
        else:
            # If no pose found in this frame, you could skip or break
            print(f"[INFO] No pose found in frame {i}.")
        # Optionally add a small delay if needed, e.g., time.sleep(0.1)

    if len(all_poses) == 0:
        print("[ERROR] Could not get any valid camera poses.")
        exit(0)

    # Average all poses from the 10 frames
    rvec_avg, tvec_avg = average_poses(all_poses)

    # For debugging, let's compute the final "camera position" in global coords
    # camera->world rotation matrix
    R_cw_final, _ = cv2.Rodrigues(rvec_avg)
    # The X, Y, Z of the camera in the global frame
    camera_position = tvec_avg
    print("\n[RESULT] AVERAGED CAMERA POSE (over 10 frames)")
    print(f"  Camera position (X,Y,Z): {camera_position[0]:.3f}, "
          f"{camera_position[1]:.3f}, {camera_position[2]:.3f}")

    # Optional: compute Euler angles from R_cw
    def rotationMatrixToEulerAngles(R):
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        if not singular:
            roll  = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw   = np.arctan2(R[1, 0], R[0, 0])
        else:
            # Gimbal lock
            roll  = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw   = 0.0
        return np.array([roll, pitch, yaw])

    eulers_rad = rotationMatrixToEulerAngles(R_cw_final)
    eulers_deg = np.degrees(eulers_rad)
    print(f"  Camera Euler Angles (deg): roll={eulers_deg[0]:.1f}, "
          f"pitch={eulers_deg[1]:.1f}, yaw={eulers_deg[2]:.1f}")

    # (Optional) Save the last annotated frame
    if annotated_frame is not None:
        # Annotate the final position
        cv2.putText(
            annotated_frame,
            f"Avg Pos: X={camera_position[0]:.1f}, Y={camera_position[1]:.1f}, Z={camera_position[2]:.1f}",
            (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2
        )
        cv2.putText(
            annotated_frame,
            f"Roll={eulers_deg[0]:.1f}, Pitch={eulers_deg[1]:.1f}, Yaw={eulers_deg[2]:.1f}",
            (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2
        )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(SAVE_FOLDER, f"camera_pose_{ts}.jpg")
        cv2.imwrite(out_path, annotated_frame)
        print(f"[INFO] Saved annotated image => {out_path}")
    else:
        print("[INFO] No annotated frame was available to save.")
