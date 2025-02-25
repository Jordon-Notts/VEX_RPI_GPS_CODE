import cv2
import numpy as np

# Define the 3D object points (tag corners in the global coordinate system)
object_points = np.array([
    [834.8662785792328, 77.38253942431052, 61.341773500095336],   # Top Left
    [830.444977628062,  -22.51392082904618, 62.41384007858043],   # Top Right
    [836.7439051026026, -23.863652699203907, -37.37845289311742],  # Bottom Right
    [841.1652060537734, 76.0328075541528,   -38.45051947160251]   # Bottom Left
], dtype=np.float32)

# Define the 2D image points (pixel coordinates)
image_points = np.array([
    [574, 306],  # Top Left
    [669, 305],  # Top Right
    [670, 400],  # Bottom Right
    [576, 401]   # Bottom Left
], dtype=np.float32)

# Camera intrinsic matrix
camera_matrix = np.array([
    [793.8411177472057, 0.0,               647.4193791910172],
    [0.0,               794.3821846338057, 364.5417378038298],
    [0.0,               0.0,               1.0]
], dtype=np.float32)

# Distortion coefficients
dist_coeffs = np.array([[-0.07927291936116608, 0.12386565928522575, 
                         0.0008165084444783896, 0.0012765443520050806, 
                        -0.07442718503493052]], dtype=np.float32)

# Solve for rotation and translation vectors using solvePnP
success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
if not success:
    raise RuntimeError("solvePnP failed to find a solution.")

# Convert rotation vector to rotation matrix (global -> camera)
R_global_to_camera, _ = cv2.Rodrigues(rvec)
print("Rotation Matrix (Global -> Camera):\n", R_global_to_camera)
print("Translation Vector (Tag in Camera coordinates):\n", tvec)

# Compute the camera position in the global coordinate system:
# Camera position (global) = -R^T * tvec
camera_position = -R_global_to_camera.T @ tvec
print("Camera Position in Global Coordinates:\n", camera_position)

# To get the camera's orientation in the global coordinate system,
# invert the rotation: R_camera_to_global = R_global_to_camera.T
R_camera_to_global = R_global_to_camera.T

# Define a function to extract Euler angles from a rotation matrix (assuming ZYX order)
def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])   # roll
        y = np.arctan2(-R[2, 0], sy)         # pitch
        z = np.arctan2(R[1, 0], R[0, 0])      # yaw
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])

# Extract Euler angles (in radians) representing camera orientation in global coordinates
euler_angles_rad = rotationMatrixToEulerAngles(R_camera_to_global)
print("Camera Euler Angles (radians) in Global Coordinates (roll, pitch, yaw):\n", euler_angles_rad)

# Convert Euler angles to degrees
euler_angles_deg = np.degrees(euler_angles_rad)
print("Camera Euler Angles (degrees) in Global Coordinates (roll, pitch, yaw):\n", euler_angles_deg.round(1))
