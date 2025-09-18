import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera
import cv2
import numpy as np
import math


def yaw_to_quat(yaw_deg):
    yaw = math.radians(yaw_deg)
    w = math.cos(yaw / 2)
    z = math.sin(yaw / 2)
    return (0.0, 0.0, z, w)

USE_AUTO_BRIGHTNESS = True

def compute_avg_brightness(frame, src_points=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if src_points is not None:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        src_poly = np.array(src_points, dtype=np.int32)
        cv2.fillPoly(mask, [src_poly], 1)
        avg_brightness = np.mean(gray[mask == 1])
    else:
        avg_brightness = np.mean(gray)
    return avg_brightness

def get_src_points(image_shape, speed=0):
    h, w = image_shape[:2]
    ref_w, ref_h = 1278, 720
    scale_w = w / ref_w
    scale_h = h / ref_h
    left_bottom  = [118, 590]
    right_bottom = [1077, 590]
    top_right    = [730, 408]
    top_left     = [519, 408]
    speed_norm = min(speed / 120.0, 1.0)
    top_shift = -40 * speed_norm
    side_shift = 100 * speed_norm
    src = np.float32([
        [left_bottom[0] * scale_w, left_bottom[1] * scale_h],
        [right_bottom[0] * scale_w, right_bottom[1] * scale_h],
        [(top_right[0] - side_shift) * scale_w, (top_right[1] + top_shift) * scale_h],
        [(top_left[0] + side_shift) * scale_w,  (top_left[1]  + top_shift) * scale_h]
    ])
    return src

def nothing(x):
    pass


cv2.namedWindow("White Mask", cv2.WINDOW_NORMAL)
cv2.resizeWindow("White Mask", 400, 300)
cv2.namedWindow("Yellow Mask", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Yellow Mask", 400, 300)
cv2.namedWindow("Shadow Mask", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Shadow Mask", 400, 300)


cv2.createTrackbar("H min", "White Mask", 51, 180, nothing)
cv2.createTrackbar("H max", "White Mask", 82, 180, nothing)
cv2.createTrackbar("S min", "White Mask", 0, 255, nothing)
cv2.createTrackbar("S max", "White Mask", 27, 255, nothing)
cv2.createTrackbar("V min", "White Mask", 51, 255, nothing)
cv2.createTrackbar("V max", "White Mask", 204, 255, nothing)

cv2.createTrackbar("H min", "Yellow Mask", 81, 180, nothing)
cv2.createTrackbar("H max", "Yellow Mask", 180, 180, nothing)
cv2.createTrackbar("S min", "Yellow Mask", 150, 255, nothing)
cv2.createTrackbar("S max", "Yellow Mask", 201, 255, nothing)
cv2.createTrackbar("V min", "Yellow Mask", 180, 255, nothing)
cv2.createTrackbar("V max", "Yellow Mask", 255, 255, nothing)

cv2.createTrackbar("H min", "Shadow Mask", 129, 180, nothing)
cv2.createTrackbar("H max", "Shadow Mask", 180, 180, nothing)
cv2.createTrackbar("S min", "Shadow Mask", 37, 255, nothing)
cv2.createTrackbar("S max", "Shadow Mask", 103, 255, nothing)
cv2.createTrackbar("V min", "Shadow Mask", 102, 255, nothing)
cv2.createTrackbar("V max", "Shadow Mask", 201, 255, nothing)


try:
    print("Initializing BeamNG...")
    beamng = BeamNGpy('localhost', 64256, home=r'C:\Users\user\Documents\beamng-tech\BeamNG.tech.v0.36.4.0')
    print("Opening BeamNG connection...")
    beamng.open()
    print("Connection opened successfully")

    scenario = Scenario('west_coast_usa', 'lane_detection_city')
    print("Creating scenario...")
    #scenario = Scenario('west_coast_usa', 'lane_detection_highway')
    vehicle = Vehicle('ego_vehicle', model='etk800', licence='JULIAN')
    #vehicle = Vehicle('Q8', model='adroniskq8', licence='JULIAN')

    # Spawn positions rotation conversion
    rot_city = yaw_to_quat(-133.506 + 180)
    rot_highway = yaw_to_quat(-135.678)

    # Street Spawn
    #scenario.add_vehicle(vehicle, pos=(-730.212, 94.630, 118.517), rot_quat=rot_city)

    # Highway Spawn
    print("Adding vehicle to scenario...")
    scenario.add_vehicle(vehicle, pos=(-287.210, 73.609, 112.363), rot_quat=rot_highway)

    print("Making scenario...")
    scenario.make(beamng)
    print("Loading scenario...")
    beamng.scenario.load(scenario)
    print("Starting scenario...")
    beamng.scenario.start()
    print("Scenario started!")
except Exception as e:
    print(f"ERROR during BeamNG setup: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("Setting up camera...")
    camera = Camera(
        'front_cam',
        beamng,
        vehicle,
        requested_update_time=0.01,
        is_using_shared_memory=True,
        pos=(0, -1.3, 1.4),
        dir=(0, -1, 0),
        field_of_view_y=90,
        near_far_planes=(0.1, 1000),
        resolution=(640, 360),
        is_streaming=True,
        is_render_colours=True,
    )
    print("Camera setup complete")
except Exception as e:
    print(f"ERROR during camera setup: {e}")
    import traceback
    traceback.print_exc()
    beamng.close()
    sys.exit(1)

print("Waiting for camera to provide frames...")
frame = None
max_attempts = 50
attempts = 0

while frame is None and attempts < max_attempts:
    try:
        print(f"Polling camera (attempt {attempts+1}/{max_attempts})...")
        images = camera.poll()
        if images and 'colour' in images:
            frame = np.array(images['colour'])
            print("Successfully received camera frame!")
        else:
            print("No colour image data received")
    except Exception as e:
        print(f"Error polling camera: {e}")
        frame = None
    
    attempts += 1
    cv2.waitKey(100)

if frame is None:
    print("ERROR: Could not get camera frames after multiple attempts")
    beamng.close()
    sys.exit(1)

print("Starting main loop...")
frame_counter = 0
tod_values = [0.2, 0.5, 0.7, 0.85, 0.0]
tod_names = ["Morning", "Midday", "Afternoon", "Evening", "Night"]
tod_index = 0
frames_per_tod = 200

while True:
    try:
        images = camera.poll()
        if not images or 'colour' not in images:
            print("Warning: No valid images received from camera")
            cv2.waitKey(100)
            continue

        frame = np.array(images['colour'])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        src_points = get_src_points(frame.shape, speed=0)

        if frame_counter % frames_per_tod == 0:
            tod_index = (frame_counter // frames_per_tod) % len(tod_values)
            beamng.set_tod(tod=tod_values[tod_index], play=False)
            print(f"\n--- Time of Day changed to {tod_names[tod_index]} (tod={tod_values[tod_index]}) ---\n")

        frame_counter += 1
    except Exception as e:
        print(f"Error in main loop: {e}")
        cv2.waitKey(100)
        continue

    if USE_AUTO_BRIGHTNESS:
        avg_brightness = compute_avg_brightness(frame, src_points=src_points)

        w_h_min, w_h_max = 51, 82
        w_s_min, w_s_max = 0, 27
        w_v_min, w_v_max = 51, 204

        y_h_min, y_h_max = 81, 180
        y_s_min, y_s_max = 150, 201
        y_v_min, y_v_max = 180, 255

        s_h_min, s_h_max = 129, 180
        s_s_min, s_s_max = 37, 103
        s_v_min, s_v_max = 102, 201

        # Adapt V min for white/yellow lanes based on brightness
        if avg_brightness > 190:
            adjust = int((avg_brightness - 190) * 0.8)
            w_v_min = min(w_v_min + adjust, w_v_max - 1)
            y_v_min = min(y_v_min + adjust, y_v_max - 1)
        elif avg_brightness < 50:
            adjust = int((50 - avg_brightness) * 2.5)
            w_v_min = max(w_v_min - adjust, 15)  # allow V min to go as low as 15
            y_v_min = max(y_v_min - adjust, 15)
            s_s_min = max(s_s_min - int(adjust/2.5), 0)
        elif avg_brightness < 110:
            adjust = int((110 - avg_brightness) * 2.0)
            w_v_min = max(w_v_min - adjust, 5)
            y_v_min = max(y_v_min - adjust, 5)
            s_s_min = max(s_s_min - int(adjust/2), 0)
        

        cv2.setTrackbarPos("H min", "White Mask", w_h_min)
        cv2.setTrackbarPos("H max", "White Mask", w_h_max)
        cv2.setTrackbarPos("S min", "White Mask", w_s_min)
        cv2.setTrackbarPos("S max", "White Mask", w_s_max)
        cv2.setTrackbarPos("V min", "White Mask", w_v_min)
        cv2.setTrackbarPos("V max", "White Mask", w_v_max)

        cv2.setTrackbarPos("H min", "Yellow Mask", y_h_min)
        cv2.setTrackbarPos("H max", "Yellow Mask", y_h_max)
        cv2.setTrackbarPos("S min", "Yellow Mask", y_s_min)
        cv2.setTrackbarPos("S max", "Yellow Mask", y_s_max)
        cv2.setTrackbarPos("V min", "Yellow Mask", y_v_min)
        cv2.setTrackbarPos("V max", "Yellow Mask", y_v_max)

        cv2.setTrackbarPos("H min", "Shadow Mask", s_h_min)
        cv2.setTrackbarPos("H max", "Shadow Mask", s_h_max)
        cv2.setTrackbarPos("S min", "Shadow Mask", s_s_min)
        cv2.setTrackbarPos("S max", "Shadow Mask", s_s_max)
        cv2.setTrackbarPos("V min", "Shadow Mask", s_v_min)
        cv2.setTrackbarPos("V max", "Shadow Mask", s_v_max)

        if frame_counter % 50 == 0:
            print(f"[Auto] Avg Brightness: {avg_brightness:.1f}")
            print(f"White: H({w_h_min}-{w_h_max}) S({w_s_min}-{w_s_max}) V({w_v_min}-{w_v_max})")
            print(f"Yellow: H({y_h_min}-{y_h_max}) S({y_s_min}-{y_s_max}) V({y_v_min}-{y_v_max})")
            print(f"Shadow: H({s_h_min}-{s_h_max}) S({s_s_min}-{s_s_max}) V({s_v_min}-{s_v_max})")

    else:
        w_h_min = cv2.getTrackbarPos("H min", "White Mask")
        w_h_max = cv2.getTrackbarPos("H max", "White Mask")
        w_s_min = cv2.getTrackbarPos("S min", "White Mask")
        w_s_max = cv2.getTrackbarPos("S max", "White Mask")
        w_v_min = cv2.getTrackbarPos("V min", "White Mask")
        w_v_max = cv2.getTrackbarPos("V max", "White Mask")

        y_h_min = cv2.getTrackbarPos("H min", "Yellow Mask")
        y_h_max = cv2.getTrackbarPos("H max", "Yellow Mask")
        y_s_min = cv2.getTrackbarPos("S min", "Yellow Mask")
        y_s_max = cv2.getTrackbarPos("S max", "Yellow Mask")
        y_v_min = cv2.getTrackbarPos("V min", "Yellow Mask")
        y_v_max = cv2.getTrackbarPos("V max", "Yellow Mask")

        s_h_min = cv2.getTrackbarPos("H min", "Shadow Mask")
        s_h_max = cv2.getTrackbarPos("H max", "Shadow Mask")
        s_s_min = cv2.getTrackbarPos("S min", "Shadow Mask")
        s_s_max = cv2.getTrackbarPos("S max", "Shadow Mask")
        s_v_min = cv2.getTrackbarPos("V min", "Shadow Mask")
        s_v_max = cv2.getTrackbarPos("V max", "Shadow Mask")

    if frame_counter % 30 == 0:
        print(f"White: H({w_h_min}-{w_h_max}) S({w_s_min}-{w_s_max}) V({w_v_min}-{w_v_max})")
        print(f"Yellow: H({y_h_min}-{y_h_max}) S({y_s_min}-{y_s_max}) V({y_v_min}-{y_v_max})")
        print(f"Shadow: H({s_h_min}-{s_h_max}) S({s_s_min}-{s_s_max}) V({s_v_min}-{s_v_max})")

    white_mask = cv2.inRange(hsv, np.array([w_h_min, w_s_min, w_v_min]), np.array([w_h_max, w_s_max, w_v_max]))
    yellow_mask = cv2.inRange(hsv, np.array([y_h_min, y_s_min, y_v_min]), np.array([y_h_max, y_s_max, y_v_max]))
    shadow_mask = cv2.inRange(hsv, np.array([s_h_min, s_s_min, s_v_min]), np.array([s_h_max, s_s_max, s_v_max]))

    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    combined_mask = cv2.bitwise_or(combined_mask, shadow_mask)

    if USE_AUTO_BRIGHTNESS:
        cv2.putText(frame, f"Avg Brightness: {avg_brightness:.1f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Mask", combined_mask)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cv2.destroyAllWindows()
beamng.close()
