#!/usr/bin/env python3
import cv2
import numpy as np
import json
import os
import time
from datetime import datetime
from picamera2 import Picamera2
import math

from packaging import version

USE_ARUCO_DETECTOR_CLASS = False

# Setup ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Setup parameters and decide which detection API to use
try:
    if version.parse(cv2.__version__) >= version.parse("4.7.0") and hasattr(cv2.aruco, "ArucoDetector"):
        USE_ARUCO_DETECTOR_CLASS = True
        aruco_params = cv2.aruco.DetectorParameters()
except Exception as e:
    print("⚠️ Warning: Failed to create DetectorParameters, falling back to default detection.")
    aruco_params = None

# -----------------------------------------------------------
# Configuration / Paths
# -----------------------------------------------------------
CALIBRATION_FILE = "camera_calibration.json"
MARKER_DB_FILE   = "aruco_marker_positions.json"
OUTPUT_FOLDER    = "output_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------------------------------------
# 1) Load Camera Calibration Data
# -----------------------------------------------------------
if not os.path.exists(CALIBRATION_FILE):
    raise FileNotFoundError(f"Missing {CALIBRATION_FILE}")

with open(CALIBRATION_FILE, "r") as f:
    calib_data = json.load(f)

camera_matrix = np.array(calib_data["camera_matrix"], dtype=np.float32)
dist_coeffs   = np.array(calib_data["distortion_coeffs"], dtype=np.float32)

# -----------------------------------------------------------
# 2) Load Marker Database (with known 3D marker corners)
# -----------------------------------------------------------
if not os.path.exists(MARKER_DB_FILE):
    raise FileNotFoundError(f"Missing {MARKER_DB_FILE}")

with open(MARKER_DB_FILE, "r") as f:
    marker_db = json.load(f)

# -----------------------------------------------------------
# 4) Initialize PiCamera2 (outside functions for fast capture)
# -----------------------------------------------------------
picam2 = Picamera2()
config = picam2.create_still_configuration(main={"size": (1280, 720), "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(1)  # Allow camera to warm up

# -----------------------------------------------------------
# 5) Function: Get Position from Markers (with refined corner detection)
# -----------------------------------------------------------
def get_position_from_markers():
    """
    Captures a frame, refines the detected ArUco marker corners, draws the edges
    and corners, computes the camera pose, overlays the pose information, saves the
    annotated image, and returns a tuple (cam_pos, yaw, annotated_frame).
    Returns (None, None, frame) if no markers are detected or if pose estimation fails.
    """
    frame = picam2.capture_array()
    # Optionally mirror the frame:
    # frame = cv2.flip(frame, 1)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if USE_ARUCO_DETECTOR_CLASS and aruco_params is not None:
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        if aruco_params is not None:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

    computed_pose = None
    if ids is not None and len(ids) > 0:
        # Refine corners for each detected marker.
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        refined_corners = [cv2.cornerSubPix(gray, c, (3,3), (-1,-1), criteria) for c in corners]
        
        # Draw edges and mark corners.
        for i, marker in enumerate(refined_corners):
            marker_id = str(ids[i][0])
            if marker_id in marker_db:
                pts = np.int32(marker).reshape(-1, 2)
                for pt in pts:
                    cv2.circle(frame, tuple(pt), 5, (0,255,0), -1)
                for j in range(len(pts)):
                    pt1 = tuple(pts[j])
                    pt2 = tuple(pts[(j+1) % len(pts)])
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
                center = np.mean(pts, axis=0).astype(int)
                cv2.putText(frame, f"ID {marker_id}", (center[0]-10, center[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Build arrays for pose estimation.
        all_obj_points = []
        all_img_points = []
        for i, marker in enumerate(refined_corners):
            marker_id = str(ids[i][0])
            if marker_id in marker_db:
                data = marker_db[marker_id]["corner_points"]
                corners_world = np.array([
                    [data["top_left"]["x"],     data["top_left"]["y"],     data["top_left"]["z"]],
                    [data["top_right"]["x"],    data["top_right"]["y"],    data["top_right"]["z"]],
                    [data["bottom_right"]["x"], data["bottom_right"]["y"], data["bottom_right"]["z"]],
                    [data["bottom_left"]["x"],  data["bottom_left"]["y"],  data["bottom_left"]["z"]]
                ], dtype=np.float32)
                all_obj_points.append(corners_world)
                all_img_points.append(marker.reshape(-1, 2))
        if len(all_obj_points) > 0:
            all_obj_points = np.vstack(all_obj_points)
            all_img_points = np.vstack(all_img_points)
            ret_pnp, rvec, tvec = cv2.solvePnP(all_obj_points, all_img_points,
                                               camera_matrix, dist_coeffs,
                                               flags=cv2.SOLVEPNP_ITERATIVE)
            if ret_pnp:
                computed_pose = (rvec, tvec)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 50)
                R, _ = cv2.Rodrigues(rvec)
                cam_pos = -R.T.dot(tvec)
                forward_vector = R.T.dot(np.array([[0],[0],[1]], dtype=np.float32))
                yaw = np.degrees(np.arctan2(forward_vector[1,0], forward_vector[0,0]))
                if yaw < 0:
                    yaw += 360
                
                pos_text = f"Pos: X={cam_pos[0,0]:.1f}, Y={cam_pos[1,0]:.1f}, Z={cam_pos[2,0]:.1f}"
                yaw_text = f"Yaw: {yaw:.1f} deg"
                cv2.putText(frame, pos_text, (20, frame.shape[0]-40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                cv2.putText(frame, yaw_text, (20, frame.shape[0]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    
                print("Camera Position:", cam_pos.flatten())
                print("Camera Yaw:", yaw, "deg")
    
                return cam_pos.flatten(), yaw, frame
            else:
                print("solvePnP failed.")
                return None, None, frame
        else:
            print("No known markers found for pose estimation.")
            return None, None, frame
    else:
        print("No markers detected.")
        return None, None, frame

def save_annotated_image():

    """
    Calls get_position_from_markers() to capture and annotate an image,
    and returns the camera position, yaw, and annotated frame.
    """

    cam_pos, yaw, frame = get_position_from_markers()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_image = os.path.join(OUTPUT_FOLDER, f"{timestamp}_annotated_marker_pose.jpg")
    cv2.imwrite(output_image, frame)
    print(f"Annotated image saved as {output_image}")

    return cam_pos, yaw

def make_map_image(cam_pos, yaw, background_path='lib/background_image.png'):
    """
    Generates a 2D map image showing:
      - A field boundary (3657 mm x 3657 mm centered on the map)
      - A grid.
      - The robot represented as a filled 400 mm x 400 mm square (rotated to match its heading)
        with a heading arrow.
    If a valid background image is provided, it is loaded and blended with white at 50% opacity.
    Returns the map image.
    """
    map_size = 500  # pixels
    scale = 0.1     # 1 mm = 0.1 pixel; adjust as needed
    map_center = (map_size // 2, map_size // 2)
    
    # Load background image if available, and blend with white at 50% each.
    if background_path is not None and os.path.exists(background_path):
        bg = cv2.imread(background_path, cv2.IMREAD_COLOR)
        bg = cv2.resize(bg, (map_size, map_size))
        white_img = np.ones_like(bg) * 255
        map_img = cv2.addWeighted(bg, 0.5, white_img, 0.5, 0)
    else:
        map_img = np.ones((map_size, map_size, 3), dtype=np.uint8) * 255

    # Draw grid lines.
    for i in range(0, map_size, 50):
        cv2.line(map_img, (i, 0), (i, map_size), (200, 200, 200), 1)
        cv2.line(map_img, (0, i), (map_size, i), (200, 200, 200), 1)
    
    # Draw field boundary (3657 mm x 3657 mm).
    field_size_mm = 3657
    half_field_pixels = int((field_size_mm * scale) / 2)
    top_left_field = (map_center[0] - half_field_pixels, map_center[1] - half_field_pixels)
    bottom_right_field = (map_center[0] + half_field_pixels, map_center[1] + half_field_pixels)
    cv2.rectangle(map_img, top_left_field, bottom_right_field, (0, 255, 0), 2)
    
    # Draw the robot as a filled square.
    robot_size_mm = 400.0
    half_robot_mm = robot_size_mm / 2.0

    # Define robot square in local coordinates (mm), centered at (0,0).
    square_local = np.array([
        [-half_robot_mm, -half_robot_mm],
        [ half_robot_mm, -half_robot_mm],
        [ half_robot_mm,  half_robot_mm],
        [-half_robot_mm,  half_robot_mm]
    ], dtype=np.float32)
    
    # Create a 2D rotation matrix for the yaw angle.
    yaw_rad = math.radians(yaw)
    R2 = np.array([
        [math.cos(yaw_rad), -math.sin(yaw_rad)],
        [math.sin(yaw_rad),  math.cos(yaw_rad)]
    ], dtype=np.float32)
    
    # Rotate the square using the rotation matrix (without transposing).
    rotated_square = np.dot(square_local, R2)
    rotated_square_px = rotated_square * scale  # convert mm to pixels
    
    # Convert robot position from mm to map coordinates: (map_center + (x*scale, -y*scale)).
    x_img = int(map_center[0] + cam_pos[0] * scale)
    y_img = int(map_center[1] - cam_pos[1] * scale)
    square_map = rotated_square_px + np.array([x_img, y_img])
    square_map = square_map.astype(np.int32)
    
    # Fill the robot square with a color (e.g., magenta).
    cv2.fillPoly(map_img, [square_map.reshape((-1, 1, 2))], color=(255, 0, 255))
    
    # Draw a heading arrow from the center of the square.
    arrow_length = 60  # pixels
    arrow_end = (int(x_img + arrow_length * math.cos(yaw_rad)),
                 int(y_img - arrow_length * math.sin(yaw_rad)))
    cv2.arrowedLine(map_img, (x_img, y_img), arrow_end, (255, 0, 0), 2)
    cv2.putText(map_img, "Robot", (x_img - 40, y_img - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return map_img

def save_map(cam_pos, yaw):
    """
    Generates a map image using the provided camera position and yaw,
    saves it with a timestamp, and returns the map image.
    """
    map_img = make_map_image(cam_pos, yaw)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    map_filename = os.path.join(OUTPUT_FOLDER, f"{timestamp}_map.jpg")
    cv2.imwrite(map_filename, map_img)
    print(f"Map image saved as {map_filename}")
    return map_img


def kill_camera():
    picam2.stop()
    print()
    print('PI Camara Killed')

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
if __name__ == "__main__":
    try:
        for i in range(5):
            pos, yaw = save_annotated_image()
            if pos is not None:
                print(f"\n[RESULT] Camera Position: {pos}")
                print(f"[RESULT] Camera Yaw: {yaw:.1f} deg")
                # Annotated image is already saved inside get_position_from_markers().
                save_map(pos, yaw)
            else:
                print("Failed to determine camera pose.")
            time.sleep(1)
    except KeyboardInterrupt:
        print("CTRL+C pressed. Exiting.")
    finally:
        kill_camera()
