import cv2
import numpy as np
import json
import os
from datetime import datetime
from picamera2 import Picamera2

# ✅ File Paths
CALIBRATION_FILE_JSON = "camera_calibration.json"
MARKER_LOCATIONS_JSON = "aruco_marker_positions.json"

# ✅ Load Camera Calibration Data
with open(CALIBRATION_FILE_JSON, "r") as json_file:
    calibration_data = json.load(json_file)

camera_matrix = np.array(calibration_data["camera_matrix"])
distortion_coeffs = np.array(calibration_data["distortion_coeffs"])

# ✅ Marker Size (in mm)
MARKER_SIZE_MM = 100.0

# ✅ Load stored marker positions (from JSON)
def load_marker_positions():
    if not os.path.exists(MARKER_LOCATIONS_JSON):
        print(f"⚠️ {MARKER_LOCATIONS_JSON} not found. Creating an empty file.")
        with open(MARKER_LOCATIONS_JSON, "w") as f:
            json.dump({}, f)
        return {}

    with open(MARKER_LOCATIONS_JSON, "r") as file:
        return json.load(file)

# ✅ Load ArUco Dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

# ✅ Initialize Camera
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration(main={"size": (1280, 720)}))
picam2.start()

# ✅ Save Folder
SAVE_FOLDER = "aruco_detections"
os.makedirs(SAVE_FOLDER, exist_ok=True)

def detect_camera_position():
    """Detects ArUco markers and estimates the camera position & compass yaw."""

    # Capture image
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, _ = detector.detectMarkers(gray)

    # Load stored marker positions
    stored_markers = load_marker_positions()

    if ids is not None:
        camera_positions = []
        yaws = []

        for i, marker_id in enumerate(ids):
            marker_id = str(marker_id[0])  # Convert ID to string (for JSON keys)
            marker_corners = corners[i][0]

            # Define 3D object points for the marker
            obj_points = np.array([
                [-MARKER_SIZE_MM / 2,  MARKER_SIZE_MM / 2, 0],  # Top-left
                [ MARKER_SIZE_MM / 2,  MARKER_SIZE_MM / 2, 0],  # Top-right
                [ MARKER_SIZE_MM / 2, -MARKER_SIZE_MM / 2, 0],  # Bottom-right
                [-MARKER_SIZE_MM / 2, -MARKER_SIZE_MM / 2, 0]   # Bottom-left
            ], dtype=np.float32)

            # Solve PnP (Pose Estimation)
            ret, rvec, tvec = cv2.solvePnP(obj_points, marker_corners, camera_matrix, distortion_coeffs)

            if ret:
                # ✅ Convert rotation vector to rotation matrix
                R, _ = cv2.Rodrigues(rvec)

                # ✅ Compute camera position relative to the marker
                camera_position = -np.dot(R.T, tvec)

                # ✅ Convert to the new coordinate system
                camera_x = camera_position[0][0]  # West-East (X)
                camera_y = camera_position[2][0]  # South-North (Y)
                camera_z = camera_position[1][0]  # Height (Z)

                camera_positions.append((camera_x, camera_y, camera_z))

                # ✅ Compute yaw (compass bearing)
                yaw_radians = np.arctan2(R[1, 0], R[0, 0])
                yaw_degrees = (np.degrees(yaw_radians) + 360) % 360  # Convert to 0-360° range
                yaws.append(yaw_degrees)

                # ✅ Apply stored marker position and adjust for compass bearing
                if marker_id in stored_markers:
                    stored_x = stored_markers[marker_id]["x"]
                    stored_y = stored_markers[marker_id]["y"]
                    stored_z = stored_markers[marker_id]["z"]
                    stored_bearing = stored_markers[marker_id]["yaw"]

                    # Adjust camera position using marker reference
                    final_x = stored_x + camera_x
                    final_y = stored_y + camera_y
                    final_z = stored_z + camera_z
                    adjusted_yaw = (stored_bearing + yaw_degrees) % 360

                    # ✅ **NEW FORMATTED LABEL ON THE MARKER**
                    marker_text = (
                        f"MARKER Position\n"
                        f"ID: {marker_id}\n"
                        f"x: {stored_x:.1f} mm\n"
                        f"y: {stored_y:.1f} mm\n"
                        f"z: {stored_z:.1f} mm\n"
                        f"Angle: {stored_bearing:.1f}°"
                    )

                    # Find a good position to display text (above the marker)
                    marker_center = marker_corners.mean(axis=0).astype(int)
                    text_x, text_y = marker_center[0] - 50, marker_center[1] - 20

                    for j, line in enumerate(marker_text.split("\n")):
                        cv2.putText(frame, line, (text_x, text_y + (j * 20)- 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # ✅ **NEW FORMATTED RELATIVE POSITION LABEL**
                    rel_text = (
                        f"CAMERA Position\n"
                        f"ID: {marker_id}\n"
                        f"x: {camera_x:.1f} mm\n"
                        f"y: {camera_y:.1f} mm\n"
                        f"z: {camera_z:.1f} mm\n"
                        f"Angle: {yaw_degrees:.1f}°"
                    )

                    # Display relative position text next to the marker
                    for j, line in enumerate(rel_text.split("\n")):
                        cv2.putText(frame, line, (text_x, text_y + (j * 20) + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                    # ✅ **DRAW THE TRIDENT (AXES) ON THE MARKER**
                    cv2.drawFrameAxes(frame, camera_matrix, distortion_coeffs, rvec, tvec, MARKER_SIZE_MM / 2)

        # ✅ Compute the average camera position and yaw
        if camera_positions:
            avg_x = np.mean([c[0] for c in camera_positions])
            avg_y = np.mean([c[1] for c in camera_positions])
            avg_z = np.mean([c[2] for c in camera_positions])
            avg_yaw = np.mean(yaws)

            # ✅ Overlay camera position in the top-left corner
            camera_pos_text = f"Cam Pos: {avg_x:.1f}, {avg_y:.1f}, {avg_z:.1f} mm"
            yaw_text = f"Yaw: {avg_yaw:.1f}°"

            cv2.putText(frame, camera_pos_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, yaw_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

            # ✅ Save the image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(SAVE_FOLDER, f"aruco_avg_{timestamp}.jpg")
            cv2.imwrite(save_path, frame)

            print(f"✅ Image saved: {save_path}")
            return avg_x, avg_y, avg_z, avg_yaw

    else:
        print("⚠️ No ArUco markers detected.")
        return False

if __name__ == "__main__":
    import time
    time.sleep(1)
    result = detect_camera_position()
    print(f"\n📍 Camera Position & Yaw: {result}" if result else "\n⚠️ No Marker Detected.")
