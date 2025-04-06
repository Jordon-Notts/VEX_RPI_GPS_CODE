import cv2
import numpy as np
import json
import os
from datetime import datetime
from picamera2 import Picamera2

# Load camera calibration data
CALIBRATION_FILE_JSON = "camera_calibration.json"

with open(CALIBRATION_FILE_JSON, "r") as json_file:
    calibration_data = json.load(json_file)

camera_matrix = np.array(calibration_data["camera_matrix"])
distortion_coeffs = np.array(calibration_data["distortion_coeffs"])

# ✅ Specify the real-world size of the ArUco marker (in millimeters)
MARKER_SIZE_MM = 100.0  # Adjust to match your printed marker

# ArUco Dictionary & Parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

# Initialize camera
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration(main={"size": (1280, 720)}))
picam2.start()

# ✅ Define the folder where images will be saved
SAVE_FOLDER = "aruco_detections"
os.makedirs(SAVE_FOLDER, exist_ok=True)

def detect_camera_position():
    """Detect ArUco markers and estimate the average CAMERA position and yaw."""

    # Capture image from the camera
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        camera_positions = []
        yaws = []

        for i, marker_id in enumerate(ids):
            marker_corners = corners[i][0]

            # Define real-world object points for the marker
            obj_points = np.array([
                [-MARKER_SIZE_MM / 2,  MARKER_SIZE_MM / 2, 0],  # Top-left
                [ MARKER_SIZE_MM / 2,  MARKER_SIZE_MM / 2, 0],  # Top-right
                [ MARKER_SIZE_MM / 2, -MARKER_SIZE_MM / 2, 0],  # Bottom-right
                [-MARKER_SIZE_MM / 2, -MARKER_SIZE_MM / 2, 0]   # Bottom-left
            ], dtype=np.float32)

            # Solve for rotation (rvec) and translation (tvec)
            ret, rvec, tvec = cv2.solvePnP(obj_points, marker_corners, camera_matrix, distortion_coeffs)

            if ret:
                # ✅ Convert rotation vector to rotation matrix
                R, _ = cv2.Rodrigues(rvec)

                # ✅ Compute camera position relative to the marker (C = -R^T * tvec)
                camera_position = -np.dot(R.T, tvec)
                camera_positions.append(camera_position)

                # ✅ Compute yaw (rotation around Z-axis)
                yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))  # Extract yaw from rotation matrix
                yaws.append(yaw)

                # ✅ Draw marker and axes
                cv2.aruco.drawDetectedMarkers(frame, [marker_corners.reshape((4, 1, 2))], np.array([[ids[i][0]]], dtype=np.int32))
                cv2.drawFrameAxes(frame, camera_matrix, distortion_coeffs, rvec, tvec, MARKER_SIZE_MM / 2)

                # ✅ Overlay marker position **directly over the marker**
                marker_pos_text = f"{marker_id[0]}: {tvec.ravel()[0]:.1f}, {tvec.ravel()[1]:.1f}, {tvec.ravel()[2]:.1f} mm"

                # Compute text position (above marker center)
                marker_center = marker_corners.mean(axis=0).astype(int)
                text_x, text_y = marker_center[0] - 30, marker_center[1] - 10

                cv2.putText(frame, marker_pos_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # ✅ Compute the average camera position and yaw
        if camera_positions:
            avg_camera_position = np.mean(camera_positions, axis=0)
            avg_yaw = np.mean(yaws)

            # ✅ Overlay camera position in **top-left corner**
            camera_pos_text = f"Cam Pos: {avg_camera_position.ravel()[0]:.1f}, {avg_camera_position.ravel()[1]:.1f}, {avg_camera_position.ravel()[2]:.1f} mm"
            yaw_text = f"Yaw: {avg_yaw:.1f}°"

            cv2.putText(frame, camera_pos_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, yaw_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

            # ✅ Save the image in a folder named after the marker ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(SAVE_FOLDER, f"aruco_avg_{timestamp}.jpg")
            cv2.imwrite(save_path, frame)

            print(f"✅ Image saved: {save_path}")
            print(f"📍 Average Camera Position (X, Y, Z): {avg_camera_position.ravel()}")
            print(f"🔄 Average Yaw: {avg_yaw:.1f}°")

            # ✅ Return only X, Y, and Yaw
            return avg_camera_position.ravel()[0], avg_camera_position.ravel()[1], avg_yaw

    else:
        print("⚠️ No ArUco markers detected.")
        return False

if __name__ == "__main__":
    import time

    time.sleep(1)

    result = detect_camera_position()

    if result:
        print(f"\n📍 Camera Position (X, Y) & Yaw: {result}")
    else:
        print("\n⚠️ No Marker Detected.")

    time.sleep(2)
