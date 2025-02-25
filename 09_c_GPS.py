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
SAVE_FOLDER = "camera_localization_images"
os.makedirs(SAVE_FOLDER, exist_ok=True)

NUM_FRAMES = 10  # Number of frames to average

# -----------------------------------------------------------
# 1) Load Camera Intrinsics
# -----------------------------------------------------------
if not os.path.exists(CALIBRATION_FILE_JSON):
    raise FileNotFoundError(f"Missing {CALIBRATION_FILE_JSON}")

with open(CALIBRATION_FILE_JSON, "r") as f:
    calib_data = json.load(f)

camera_matrix = np.array(calib_data["camera_matrix"], dtype=np.float32)  # shape (3,3)
dist_coeffs   = np.array(calib_data["distortion_coeffs"], dtype=np.float32)  # shape (k,)

# -----------------------------------------------------------
# 2) Load Known Marker Corners in the World
# -----------------------------------------------------------
if not os.path.exists(MARKER_LOCATIONS_JSON):
    raise FileNotFoundError(f"Missing {MARKER_LOCATIONS_JSON}")

with open(MARKER_LOCATIONS_JSON, "r") as f:
    marker_db = json.load(f)
# marker_db is expected to be in a form such as:
# {
#    "23": {
#       "corner_points": {
#           "top_left":  {"x": Xtl, "y": Ytl, "z": Ztl},
#           "top_right": {"x": Xtr, "y": Ytr, "z": Ztr},
#           "bottom_right": {"x": Xbr, "y": Ybr, "z": Zbr},
#           "bottom_left":  {"x": Xbl, "y": Ybl, "z": Zbl}
#       }
#    },
#    ...
# }

def get_marker_world_corners(marker_id):
    """
    Returns a (4,3) np.array of the known marker corners in the world frame,
    in the same order that ArUco typically uses:
      [top_left, top_right, bottom_right, bottom_left].
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
# 4) Average Poses Utility
# -----------------------------------------------------------
def average_poses(rvec_tvec_list):
    """
    Naive averaging of rotation + translation:
      1) Convert each rvec->matrix, sum them
      2) Re-orthonormalize via SVD
      3) Average translations
    Returns (rvec, tvec) as np.array(3,) each.
    """
    R_sum = np.zeros((3,3), dtype=np.float64)
    t_sum = np.zeros((3,1), dtype=np.float64)

    for (rvec, tvec) in rvec_tvec_list:
        R_i, _ = cv2.Rodrigues(rvec)  # axis-angle to 3x3
        R_sum += R_i
        t_sum += tvec.reshape(3,1)

    N = len(rvec_tvec_list)
    R_avg = R_sum / N
    t_avg = t_sum / N

    # Orthonormalize R_avg
    U, _, Vt = np.linalg.svd(R_avg)
    R_ortho = U @ Vt

    rvec_out, _ = cv2.Rodrigues(R_ortho)
    return rvec_out.flatten(), t_avg.flatten()

# -----------------------------------------------------------
# 5) Capture One Frame, Solve for Camera Pose
# -----------------------------------------------------------
def detect_camera_pose():
    """
    Captures a single image, detects any known markers.
    For each known marker, runs solvePnP (world->camera) and inverts to get camera->world.
    If multiple markers are detected, averages them to get a single pose (camera->world).
    Also annotates the image with the marker edges and the marker's global center.
    Returns (rvec_cw, tvec_cw, annotated_frame) or (None, None, frame) if no valid detection.
    """
    frame = picam2.capture_array()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners_2d, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        print("[INFO] No markers detected in this frame.")
        return None, None, frame

    # We'll collect all camera->world poses from any known markers.
    camera_poses_cw = []

    for i, marker_id_arr in enumerate(ids):
        marker_id_str = str(marker_id_arr[0])
        if marker_id_str not in marker_db:
            # Not in our known marker database
            continue

        # Detected 2D corners (shape (4,2))
        corners_img_2d = corners_2d[i][0]
        # Get the known 3D world corners for the marker
        corners_world_3d = get_marker_world_corners(marker_id_str)

        # Use solvePnP with known world points and detected image points
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

        # Save the computed camera->world pose for averaging later
        camera_poses_cw.append((cv2.Rodrigues(R_cw)[0], t_cw))

        # --------- Annotation on the image -----------
        # Draw the detected marker edges as a blue polygon
        pts = np.int32(corners_img_2d).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

        # Draw the coordinate axes on the marker for further illustration
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec_wc, tvec_wc, 50)

        # Label the marker ID near its center
        cxy = corners_img_2d.mean(axis=0).astype(int)
        cv2.putText(frame, f"ID {marker_id_str}", (cxy[0]-10, cxy[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Compute the global center of the marker from its known corners
        marker_world_corners = get_marker_world_corners(marker_id_str)
        marker_center = np.mean(marker_world_corners, axis=0)
        global_text = f"Glob: X={marker_center[0]:.1f}, Y={marker_center[1]:.1f}, Z={marker_center[2]:.1f}"
        # Overlay the global coordinates above the top-left corner of the detected marker
        pt_text = (pts[0][0][0], pts[0][0][1] - 20)
        cv2.putText(frame, global_text, pt_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    if len(camera_poses_cw) == 0:
        print("[INFO] Markers detected, but none matched the DB.")
        return None, None, frame

    # If multiple known markers are in view, average the resulting camera->world poses
    if len(camera_poses_cw) > 1:
        rvec_cw, tvec_cw = average_poses(camera_poses_cw)
    else:
        rvec_cw, tvec_cw = camera_poses_cw[0][0], camera_poses_cw[0][1]

    return rvec_cw, tvec_cw, frame

# -----------------------------------------------------------
# 6) Main: Capture 10 frames, average, show final result
# -----------------------------------------------------------
def main():
    all_poses = []
    last_frame = None

    for i in range(NUM_FRAMES):
        print(f"[INFO] Capturing frame {i+1}/{NUM_FRAMES} ...")
        rvec_cw, tvec_cw, frame = detect_camera_pose()
        if rvec_cw is not None:
            all_poses.append((rvec_cw, tvec_cw))
            last_frame = frame
        else:
            print("[WARN] No valid pose in this frame. Skipping.")

    if len(all_poses) == 0:
        print("[ERROR] Could not find any valid camera poses in the captured frames.")
        return

    # Average across all frames
    rvec_avg, tvec_avg = average_poses(all_poses)

    # Print the final camera position (in the world)
    print("\n[RESULT] AVERAGED CAMERA POSE")
    print("  Camera position (X, Y, Z) in world:")
    print(f"    X={tvec_avg[0]:.2f}, Y={tvec_avg[1]:.2f}, Z={tvec_avg[2]:.2f}")

    # Convert rvec_avg to a rotation matrix and compute Euler angles
    R_cw, _ = cv2.Rodrigues(rvec_avg)
    def rotationMatrixToEulerAngles(R):
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        if not singular:
            roll  = np.arctan2(R[2, 1], R[2, 2])  # around X
            pitch = np.arctan2(-R[2, 0], sy)       # around Y
            yaw   = np.arctan2(R[1, 0], R[0, 0])   # around Z
        else:
            roll  = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw   = 0.0
        return np.array([roll, pitch, yaw], dtype=np.float64)

    eulers_rad = rotationMatrixToEulerAngles(R_cw)
    eulers_deg = np.degrees(eulers_rad)
    print(f"  Camera orientation (deg): roll={eulers_deg[0]:.1f}, pitch={eulers_deg[1]:.1f}, yaw={eulers_deg[2]:.1f}")

    # Annotate the last captured frame with final camera pose information
    if last_frame is not None:
        annotated = last_frame.copy()
        cv2.putText(annotated,
                    f"Pos: X={tvec_avg[0]:.1f}, Y={tvec_avg[1]:.1f}, Z={tvec_avg[2]:.1f}",
                    (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        cv2.putText(annotated,
                    f"Roll={eulers_deg[0]:.1f}, Pitch={eulers_deg[1]:.1f}, Yaw={eulers_deg[2]:.1f}",
                    (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(SAVE_FOLDER, f"camera_pose_{ts}.jpg")
        cv2.imwrite(out_path, annotated)
        print(f"[INFO] Annotated image saved as {out_path}")

if __name__ == "__main__":
    main()
