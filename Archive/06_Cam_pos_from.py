import cv2
import numpy as np

def compute_camera_global_pose(camera_to_tag_rvec, camera_to_tag_tvec, global_tag_rvec, global_tag_tvec):
    """
    Compute the global position and orientation of the camera based on the marker's known global pose.

    Args:
        camera_to_tag_rvec (array-like): Rotation vector from camera to tag (3x1).
        camera_to_tag_tvec (array-like): Translation vector from camera to tag (3x1).
        global_tag_rvec (array-like): Rotation vector representing tag's orientation in the global frame (3x1).
        global_tag_tvec (array-like): Translation vector representing tag's position in the global frame (3x1).

    Returns:
        tuple: (camera_global_rvec, camera_global_tvec)
            - camera_global_rvec: Camera rotation vector in the global coordinate system.
            - camera_global_tvec: Camera translation vector in the global coordinate system.
    """
    # Ensure all inputs are numpy arrays with float32 type and proper shapes
    camera_to_tag_rvec = np.array(camera_to_tag_rvec, dtype=np.float32).reshape(3, 1)
    camera_to_tag_tvec = np.array(camera_to_tag_tvec, dtype=np.float32).reshape(3, 1)
    global_tag_rvec    = np.array(global_tag_rvec,    dtype=np.float32).reshape(3, 1)
    global_tag_tvec    = np.array(global_tag_tvec,    dtype=np.float32).reshape(3, 1)

    # Convert rotation vectors to rotation matrices
    R_camera_to_tag, _ = cv2.Rodrigues(camera_to_tag_rvec)
    R_global_tag, _    = cv2.Rodrigues(global_tag_rvec)

    # Compose the rotations to get the camera's global rotation:
    # R_camera_global = R_global_tag * R_camera_to_tag
    R_camera_global = R_global_tag @ R_camera_to_tag
    camera_global_rvec, _ = cv2.Rodrigues(R_camera_global)  # convert back to rotation vector

    # Compose the translations to get the camera's global translation:
    # camera_global_tvec = global_tag_tvec + R_global_tag * camera_to_tag_tvec
    camera_global_tvec = global_tag_tvec + (R_global_tag @ camera_to_tag_tvec)

    return camera_global_rvec.ravel().tolist(), camera_global_tvec.ravel().tolist()

if __name__ == "__main__":
    # -------------------------------
    # Camera-to-Tag Pose (Local Frame)
    # -------------------------------
    # The camera is head on with the tag: no rotation and 800 mm offset along the tag's z-axis.
    camera_to_tag_rvec = [0, 0, 0]      # No rotation
    camera_to_tag_tvec = [0, 0, 800]      # 800 mm along the tag's z-axis

    # -------------------------------
    # Tag-to-Global Pose (Global Frame)
    # -------------------------------
    # The tag's local coordinate system has its "up" along the y-axis.
    # To have the tag's up align with the global z-axis, we rotate the tag's coordinate system
    # by -90° about the x-axis (which maps local y --> global z).
    theta = -np.pi / 2  # -90 degrees in radians

    # Create the rotation matrix for a rotation about the x-axis by -90°:
    R_global_tag = np.array([
        [1,            0,             0],
        [0,  np.cos(theta), -np.sin(theta)],
        [0,  np.sin(theta),  np.cos(theta)]
    ])

    # Convert this rotation matrix to a rotation vector using cv2.Rodrigues.
    global_tag_rvec, _ = cv2.Rodrigues(R_global_tag)
    global_tag_rvec = global_tag_rvec.ravel().tolist()  # Flatten for our function

    # Define the tag's global translation (position in the global frame)
    # Adjust these values as needed. For example, here the tag is positioned at (-800, 0, 0).
    global_tag_tvec = [-800, 0, 0]

    # -------------------------------
    # Compute the Camera's Global Pose
    # -------------------------------
    camera_global_rvec, camera_global_tvec = compute_camera_global_pose(
        camera_to_tag_rvec, camera_to_tag_tvec,
        global_tag_rvec, global_tag_tvec
    )

    # For easier interpretation, convert the rotation vector from radians to degrees.
    camera_global_rvec_deg = np.degrees(camera_global_rvec)

    print("📍 Camera Global Rotation Vector (radians):", camera_global_rvec)
    print("📍 Camera Global Rotation Vector (degrees):", camera_global_rvec_deg.tolist())
    print("📍 Camera Global Translation Vector (mm):", camera_global_tvec)
