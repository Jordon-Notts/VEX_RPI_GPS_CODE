import cv2
import json
import numpy as np
import os
from picamera2 import Picamera2
from datetime import datetime

CALIBRATION_FILE_JSON = "camera_calibration.json"
MARKER_LOCATIONS_JSON = "aruco_marker_positions.json"
SAVE_FOLDER = "maps_on_images"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# We define the camera's current known position & yaw in mm and degrees
camera_x = 0
camera_y = 500.0
camera_yaw_deg = 90.0  # e.g. facing east

def draw_mini_map(tags, camera_x, camera_y, camera_yaw_deg,
                  map_size=(200, 200),
                  scale=0.5,
                  offset=(0, 0)):

    w, h = map_size
    map_img = np.zeros((h, w, 3), dtype=np.uint8)

    def world_to_map(wx, wy):
        mx = (wx - offset[0]) * scale
        my = (wy - offset[1]) * scale
        row = int(h - my)  # flip y
        col = int(mx)
        return (col, row)

    # Draw tags
    for (tx, ty, tag_id) in tags:
        (cx, cy) = world_to_map(tx, ty)
        cv2.circle(map_img, (cx, cy), 4, (0,255,0), -1)
        cv2.putText(map_img, str(tag_id), (cx+5, cy-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
    
    # Draw camera
    (cam_cx, cam_cy) = world_to_map(camera_x, camera_y)
    cv2.circle(map_img, (cam_cx, cam_cy), 4, (255,0,0), -1)

    # Yaw arrow
    import math
    length = 20
    rad = math.radians(camera_yaw_deg)
    dx = length * math.cos(rad)
    dy = length * math.sin(rad)
    arrow_end = (int(cam_cx + dx), int(cam_cy - dy))
    cv2.arrowedLine(map_img, (cam_cx, cam_cy), arrow_end, (255,0,0), 2)

    return map_img

def main():
    # 1) Load marker positions
    if not os.path.exists(MARKER_LOCATIONS_JSON):
        print("[ERROR] No marker file found.")
        return
    with open(MARKER_LOCATIONS_JSON, "r") as f:
        marker_db = json.load(f)

    # Extract as a list of (x, y, id)
    tag_positions = []
    for k,v in marker_db.items():
        tag_positions.append((v["x"], v["y"], k))

    # 2) Start camera, capture frame
    picam2 = Picamera2()
    picam2.configure(picam2.create_still_configuration(main={"size": (1280, 720)}))
    picam2.start()
    frame = picam2.capture_array()

    # 3) Build the mini-map
    # Decide a scale. Suppose 1 pixel = 1 mm => scale=1.0
    # If your x=1000,y=500 is too large for a 200×200, we do offset or smaller scale
    scale = 0.2  # e.g. 1 pixel = 5 mm
    offset = (0, 0)  # subtract these from each coordinate
    mini_map = draw_mini_map(
        tags=tag_positions,
        camera_x=camera_x,
        camera_y=camera_y,
        camera_yaw_deg=camera_yaw_deg,
        map_size=(200,200),
        scale=scale,
        offset=offset
    )

    # 4) Overlay top-left
    mh, mw, _ = mini_map.shape
    roi = frame[0:mh, 0:mw]  # region-of-interest
    alpha = 0.8
    cv2.addWeighted(mini_map, alpha, roi, 1-alpha, 0, roi)

    # 5) Save final
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(SAVE_FOLDER, f"map_overlay_{ts}.jpg")
    cv2.imwrite(out_path, frame)
    print(f"[INFO] Saved image => {out_path}")

if __name__ == "__main__":
    main()
