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
    frame = picam2.capture_array()
  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
  
    corners, ids, _ = detector.detectMarkers(gray)
  
    stored_markers = load_marker_positions()
  
    
    if ids is not None:
        camera_positions, yaws = [], []
        
        for i, marker_id in enumerate(ids.flatten()):
            marker_id = str(marker_id)
            marker_corners = corners[i][0]

            obj_points = np.array([
                [-MARKER_SIZE_MM / 2,  MARKER_SIZE_MM / 2, 0],
                [ MARKER_SIZE_MM / 2,  MARKER_SIZE_MM / 2, 0],
                [ MARKER_SIZE_MM / 2, -MARKER_SIZE_MM / 2, 0],
                [-MARKER_SIZE_MM / 2, -MARKER_SIZE_MM / 2, 0]
            ], dtype=np.float32)

            ret, rvec, tvec = cv2.solvePnP(obj_points, marker_corners, camera_matrix, distortion_coeffs)

            if ret:
                R, _ = cv2.Rodrigues(rvec)
                camera_position = -np.dot(R.T, tvec)
                camera_x, camera_y, camera_z = camera_position.ravel()
                yaw_degrees = (np.degrees(np.arctan2(R[1, 0], R[0, 0])) + 360) % 360

                if marker_id in stored_markers:
                    stored_x = stored_markers[marker_id]["x"]
                    stored_y = stored_markers[marker_id]["y"]
                    stored_z = stored_markers[marker_id]["z"]
                    stored_bearing = stored_markers[marker_id]["yaw"]
                    
                    # ✅ **Apply Rotation for Global Coordinates**
                    theta = np.radians(stored_bearing)
                    rot_matrix = np.array([
                        [np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta),  np.cos(theta), 0],
                        [0, 0, 1]
                    ])
                    rotated_coords = rot_matrix @ np.array([camera_x, camera_y, camera_z])
                    
                    final_x = stored_x + rotated_coords[0]
                    final_y = stored_y + rotated_coords[1]
                    final_z = stored_z + rotated_coords[2]
                    adjusted_yaw = (stored_bearing + yaw_degrees) % 360
                    
                    marker_text = f"ID:{marker_id}\nx:{stored_x:.1f}mm\ny:{stored_y:.1f}mm\nz:{stored_z:.1f}mm\nAngle:{stored_bearing:.1f}°"
                    rel_text = f"CAM x:{final_x:.1f}mm\ny:{final_y:.1f}mm\nz:{final_z:.1f}mm\nAngle:{adjusted_yaw:.1f}°"
                    
                    marker_center = marker_corners.mean(axis=0).astype(int)
                    text_x, text_y = marker_center[0] - 50, marker_center[1] - 20
                    
                    for j, line in enumerate(marker_text.split("\n")):
                        cv2.putText(frame, line, (text_x, text_y + (j * 20) - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    for j, line in enumerate(rel_text.split("\n")):
                        cv2.putText(frame, line, (text_x, text_y + (j * 20) + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    
                    cv2.drawFrameAxes(frame, camera_matrix, distortion_coeffs, rvec, tvec, MARKER_SIZE_MM / 2)

        if camera_positions:
            avg_x = np.mean([c[0] for c in camera_positions])
            avg_y = np.mean([c[1] for c in camera_positions])
            avg_z = np.mean([c[2] for c in camera_positions])
            avg_yaw = np.mean(yaws)
            
            cv2.putText(frame, f"Cam Pos: {avg_x:.1f}, {avg_y:.1f}, {avg_z:.1f} mm", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, f"Yaw: {avg_yaw:.1f}°", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(SAVE_FOLDER, f"aruco_avg_{timestamp}.jpg")
            cv2.imwrite(save_path, frame)
            print(f"✅ Image saved: {save_path}")
            return avg_x, avg_y, avg_z, avg_yaw
    
    print("⚠️ No ArUco markers detected.")
    return False

if __name__ == "__main__":
    import time
    time.sleep(1)
    result = detect_camera_position()
    print(f"\n📍 Camera Position & Yaw: {result}" if result else "\n⚠️ No Marker Detected.")
