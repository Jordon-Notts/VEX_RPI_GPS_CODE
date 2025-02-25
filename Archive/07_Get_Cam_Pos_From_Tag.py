import cv2
import numpy as np
import json
import os
from datetime import datetime
from picamera2 import Picamera2

CALIBRATION_FILE_JSON = "camera_calibration.json"
MARKER_LOCATIONS_JSON = "aruco_marker_positions.json"

SAVE_FOLDER = "aruco_detections"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# --- Load Camera Calibration ---
with open(CALIBRATION_FILE_JSON, "r") as f:
    calibration_data = json.load(f)
camera_matrix = np.array(calibration_data["camera_matrix"])
distortion_coeffs = np.array(calibration_data["distortion_coeffs"])

# --- ArUco Dictionary & Detector ---
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

# --- Initialize camera ---
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration(main={"size": (1280, 720)}))
picam2.start()

# --- Load Marker Poses (with per-marker size) ---
def load_marker_positions():
    if not os.path.exists(MARKER_LOCATIONS_JSON):
        print(f"⚠️ {MARKER_LOCATIONS_JSON} not found. Creating an empty file.")
        with open(MARKER_LOCATIONS_JSON, "w") as f:
            json.dump({}, f)
        return {}
    with open(MARKER_LOCATIONS_JSON, "r") as file:
        return json.load(file)

def compute_camera_global_pose(rvec_camera_to_aruco, 
                               tvec_camera_to_aruco, 
                               rvec_aruco_in_global, 
                               tvec_aruco_in_global):
    """
    Returns rvec_camera_in_global, tvec_camera_in_global.
    """
    rvec_camera_to_aruco = np.array(rvec_camera_to_aruco, dtype=np.float32)
    tvec_camera_to_aruco = np.array(tvec_camera_to_aruco, dtype=np.float32)
    rvec_aruco_in_global = np.array(rvec_aruco_in_global, dtype=np.float32)
    tvec_aruco_in_global = np.array(tvec_aruco_in_global, dtype=np.float32)

    R_camera_to_aruco, _ = cv2.Rodrigues(rvec_camera_to_aruco)
    R_aruco_in_global, _ = cv2.Rodrigues(rvec_aruco_in_global)

    # R_camera_in_global = R_aruco_in_global * R_camera_to_aruco
    R_camera_in_global = R_aruco_in_global @ R_camera_to_aruco
    rvec_camera_in_global, _ = cv2.Rodrigues(R_camera_in_global)

    # t_camera_in_global = t_aruco_in_global + R_aruco_in_global * t_camera_to_aruco
    t_camera_in_global = tvec_aruco_in_global + (R_aruco_in_global @ tvec_camera_to_aruco)

    return rvec_camera_in_global, t_camera_in_global

def detect_camera_position():
    # Capture image
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, _ = detector.detectMarkers(gray)

    # Load stored marker info (including size)
    stored_markers = load_marker_positions()

    camera_global_positions = []
    camera_global_yaws = []

    if ids is not None:
        for i, marker_id_arr in enumerate(ids):
            marker_id = str(marker_id_arr[0])  # e.g. '0' or '1'
            marker_corners = corners[i][0]

            if marker_id not in stored_markers:
                # No known global pose for this marker
                continue

            # --- Grab marker info from JSON ---
            marker_info = stored_markers[marker_id]
            rvec_aruco_in_global = marker_info["rvec_aruco_in_global"]
            tvec_aruco_in_global = marker_info["tvec_aruco_in_global"]
            marker_size_mm = marker_info["marker_size_mm"]

            # Build 3D object points based on the marker_size_mm
            half_size = marker_size_mm / 2.0
            obj_points = np.array([
                [-half_size,  half_size, 0],
                [ half_size,  half_size, 0],
                [ half_size, -half_size, 0],
                [-half_size, -half_size, 0]
            ], dtype=np.float32)

            # Solve PnP to get camera->marker
            success, rvec_camera_to_aruco, tvec_camera_to_aruco = cv2.solvePnP(
                obj_points, marker_corners, camera_matrix, distortion_coeffs
            )
            if not success:
                continue

            # Compute camera pose in global coords
            rvec_camera_in_global, tvec_camera_in_global = compute_camera_global_pose(
                rvec_camera_to_aruco, 
                tvec_camera_to_aruco, 
                rvec_aruco_in_global, 
                tvec_aruco_in_global
            )

            # Extract a Yaw angle for debugging (assuming Z is "up")
            R_camera_in_global, _ = cv2.Rodrigues(rvec_camera_in_global)
            yaw_rad = np.arctan2(R_camera_in_global[1, 0], R_camera_in_global[0, 0])
            yaw_deg = (np.degrees(yaw_rad) + 360) % 360

            camera_global_positions.append(tvec_camera_in_global)
            camera_global_yaws.append(yaw_deg)

            # Optional: draw the axes on the detected marker
            cv2.drawFrameAxes(frame, camera_matrix, distortion_coeffs,
                              rvec_camera_to_aruco, tvec_camera_to_aruco,
                              marker_size_mm * 0.5)

        if camera_global_positions:
            # Average pose over all detected markers
            camera_global_positions = np.vstack(camera_global_positions)
            avg_pos = camera_global_positions.mean(axis=0)
            avg_yaw = np.mean(camera_global_yaws)

            text_pos = f"Cam Position: ({avg_pos[0]:.1f}, {avg_pos[1]:.1f}, {avg_pos[2]:.1f})"
            text_yaw = f"Yaw: {avg_yaw:.1f}°"
            
            cv2.putText(frame, text_pos, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, text_yaw, (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            # Save annotated image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(SAVE_FOLDER, f"aruco_{timestamp}.jpg")
            cv2.imwrite(save_path, frame)

            print(f"✅ Saved detection image: {save_path}")
            return (avg_pos[0], avg_pos[1], avg_pos[2], avg_yaw)

    print("⚠️ No valid ArUco markers (with known global pose) detected.")
    return None

if __name__ == "__main__":
    result = detect_camera_position()
    if result:
        x, y, z, yaw = result
        print(f"\n📍 Camera Global => x: {x:.1f}, y: {y:.1f}, z: {z:.1f}, yaw: {yaw:.1f}°")
    else:
        print("\n⚠️ No global pose computed.")

