#!/usr/bin/env python3
import requests
import time

def GET_position_from_server(url):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            print("Error: Received status code", response.status_code)
            return

        data = response.json()
        # Expected JSON structure: {"id": 1, "global_position": [x, y, z], "yaw": yaw_deg}
        x = data["global_position"][0]
        y = data["global_position"][1]
        yaw = data["yaw"]

        # Ensure the yaw angle is non-negative (if negative, adjust by adding 360°)
        if yaw < 0:
            yaw += 360.0

        # Convert x, y to integers; convert yaw to tenths of a degree (as an integer)
        
        x = int(round(x,1))
        y = int(round(y,1))

        angle = int(round(yaw,1))

        return x, y, angle

    except Exception as e:
        print("Error in GET_position_from_server:", e)
        return False

# -------------------------
# Main Loop
# -------------------------

if __name__ == '__main__':

    URL = "http://yourserver:port/yourendpoint"  # Replace with your actual URL

    result = GET_position_from_server(URL)
    if result:
        x, y, angle = result
        print(f"Received Position: X={x}, Y={y}, Angle={angle/10.0:.1f} deg")
    else:
        print("Failed to get position from server.")

