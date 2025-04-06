import cv2
import numpy as np
import json
import os
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
    """Detect ArUco markers and estimate CAMERA position relative to the marker."""

    # Capture image from the camera
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
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

                print(f"\n🔍 Marker ID: {marker_id[0]}")
                # print(f"📍 Camera Position (X, Y, Z) relative to marker (in mm): {camera_position.ravel()}")
                print(f"📍 Marker Position (X, Y, Z) relative to camera (in mm): {tvec.ravel()}")
                print(f"🔄 Marker Rotation Vector: {rvec.ravel()}")

                # ✅ Ensure marker corners have the correct shape (4,1,2) and IDs are integers
                cv2.aruco.drawDetectedMarkers(frame, [marker_corners.reshape((4, 1, 2))], np.array([[ids[i][0]]], dtype=np.int32))

                # Draw the axis on the marker
                cv2.drawFrameAxes(frame, camera_matrix, distortion_coeffs, rvec, tvec, MARKER_SIZE_MM / 2)  

                # ✅ Overlay text with camera & marker position
                marker_pos_text = f"Marker Pos: {tvec.ravel()[0]:.1f}, {tvec.ravel()[1]:.1f}, {tvec.ravel()[2]:.1f} mm"
                camera_pos_text = f"Camera Pos: {camera_position.ravel()[0]:.1f}, {camera_position.ravel()[1]:.1f}, {camera_position.ravel()[2]:.1f} mm"

                cv2.putText(frame, marker_pos_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, camera_pos_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # ✅ Save the image in a folder named after the marker ID
                marker_folder = os.path.join(SAVE_FOLDER, f"marker_{marker_id[0]}")
                os.makedirs(marker_folder, exist_ok=True)
                save_path = os.path.join(marker_folder, f"aruco_{marker_id[0]}_{int(tvec[2][0])}mm.jpg")

                cv2.imwrite(save_path, frame)
                print(f"✅ Image saved: {save_path}")

    else:
        print("⚠️ No ArUco markers detected.")

if __name__ == "__main__":
    import time

    time.sleep(1)

    detect_camera_position()

    time.sleep(2)
