import cv2
import time
import os
import numpy as np
import json
from picamera2 import Picamera2

# Set parameters
CHESSBOARD_SIZE = (9, 6)  # Adjust based on your board
PREVIEW_FOLDER = "chessboard_preview"
CALIBRATION_FILE_JSON = "camera_calibration.json"
CALIBRATION_FILE_NPZ = "camera_calibration.npz"
MAX_IMAGES = 50
MIN_TIME_BETWEEN_CAPTURES = 0.001  # Minimum 0.2 seconds between captures

# Ensure output directories exist
os.makedirs(PREVIEW_FOLDER, exist_ok=True)

# Initialize camera
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration(main={"size": (1280, 720)}))  # Lower resolution for faster processing
picam2.start()

# Prepare object points (3D coordinates of chessboard corners)
square_size = 20.0  # Square size in mm

objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * square_size


# Lists to store detected corners
obj_points = []  # 3D points in real world
img_points = []  # 2D points in image plane

# Start capturing
image_count = 0
last_capture_time = time.time()

print("📸 Monitoring camera for chessboard... Press 'CTRL+C' to quit.")

while image_count < MAX_IMAGES:
    # Capture a frame from the camera
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Try to detect chessboard
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if ret:
        current_time = time.time()

        # Ensure at least 0.2 seconds between captures
        if current_time - last_capture_time >= MIN_TIME_BETWEEN_CAPTURES:
            preview_filename = os.path.join(PREVIEW_FOLDER, f"preview_{image_count:03d}.jpg")

            # Refine corners for better accuracy
            refined_corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )

            # Store points for calibration
            obj_points.append(objp)
            img_points.append(refined_corners)

            # Save a small preview image with drawn corners
            frame_with_corners = frame.copy()
            cv2.drawChessboardCorners(frame_with_corners, CHESSBOARD_SIZE, refined_corners, ret)
            cv2.imwrite(preview_filename, frame_with_corners)

            print(f"✅ Chessboard detected! Image saved: {preview_filename}")

            image_count += 1
            last_capture_time = current_time

print("\n📸 Capture complete. Proceeding to camera calibration...\n")
picam2.stop()

# Perform camera calibration
if len(obj_points) > 5:
    ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )

    if ret:
        print("\n📐 Camera Calibration Complete:")
        print("📷 Camera Matrix:\n", camera_matrix)
        print("📏 Distortion Coefficients:\n", distortion_coeffs.ravel())

        # Save calibration data in JSON format for easy access
        calibration_data = {
            "camera_matrix": camera_matrix.tolist(),
            "distortion_coeffs": distortion_coeffs.tolist(),
        }
        with open(CALIBRATION_FILE_JSON, "w") as json_file:
            json.dump(calibration_data, json_file, indent=4)

        # Also save in NumPy format for direct loading in OpenCV
        np.savez(CALIBRATION_FILE_NPZ, camera_matrix=camera_matrix, distortion_coeffs=distortion_coeffs)

        print(f"\n💾 Calibration data saved to '{CALIBRATION_FILE_JSON}' and '{CALIBRATION_FILE_NPZ}'.")

    else:
        print("⚠️ Calibration failed. Not enough valid chessboard detections.")
else:
    print("⚠️ Not enough images detected for calibration.")