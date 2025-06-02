import cv2 as cv
import numpy as np
import os

# Function to preprocess including color space conversion and masking the frames for lane detection 

def binary_conversion(warped_image):
    gray = cv.cvtColor(warped_image, cv.COLOR_BGR2GRAY)
    print("Gray min/max:", np.min(gray), np.max(gray))
    gray_normalized = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX)

    _, white_binary = cv.threshold(gray_normalized, 140, 255, cv.THRESH_BINARY)

    hsv = cv.cvtColor(warped_image, cv.COLOR_BGR2HSV)
    yellow_mask = cv.inRange(hsv, (15, 60, 60), (40, 255, 255))

    binary = cv.bitwise_or(white_binary, yellow_mask)

    kernel = np.ones((7, 7), np.uint8)
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
    return binary

# Function to apply perspective transform to the image to get a bird's eye view

def perspective_transform(image):
    height, width = image.shape[:2]
    src = np.float32([
        [width * 0.4, height * 0.65],  # Top left
        [width * 0.6, height * 0.65],  # Top right
        [width * 0.2, height * 0.9],   # Bottom left
        [width * 0.8, height * 0.9]    # Bottom right
    ])
    
    dst = np.float32([
        [width * 0.25, 0],             # Top left
        [width * 0.75, 0],             # Top right
        [width * 0.25, height],        # Bottom left
        [width * 0.75, height]         # Bottom right
    ])

    matrix = cv.getPerspectiveTransform(src, dst)
    warped = cv.warpPerspective(image, matrix, (width, height))
    return warped, matrix

# Function to define the region of interest (ROI) for lane detection

def roi(image):
    height, width = image.shape[:2]
    mask = np.zeros_like(image)
    polygon = np.array([[
        (0, height),      # Bottom left
        (0.2 * width, 0.65 * height),       # Mid left
        (0.4 * width, 0.55 * height),       # Top left
        (0.55 * width, 0.55 * height),      # Top right
        (0.65 * width, 0.7 * height),       # Mid right
        (0.9 * width, height)        # Bottom right
    ]], dtype=np.float32).astype(np.int32)

    cv.fillPoly(mask, polygon, 255)
    masked_img = cv.bitwise_and(image, mask)
    return masked_img, polygon

# Function to filter detected lines into left and right lanes based on their slopes and positions

def filter_lanes(lines, image_width):
    if lines is None:
        return None, None
    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue

        slope = (y2 - y1) / (x2 - x1)
        length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        mid_x = (x1 + x2) / 2

        if abs(slope) > 0.1 and abs(slope) < 2.0 and length > 30:
            if slope < 0 and mid_x < image_width * 0.5:
                left_lines.append(line)
            elif slope > 0 and mid_x > image_width * 0.5:
                right_lines.append(line)

    return left_lines, right_lines

# Function to convert line parameters (slope and intercept) into coordinates for drawing lines on the image

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * 0.65)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

prev_right_fit_average = np.array([-0.5, 300])
prev_left_fit_average = np.array([0.5, -50])

prev_left_curve = None
prev_left_degree = 1
prev_right_curve = None
prev_right_degree = 1

prev_left_base = None
prev_right_base = None
lane_history_size = 10
left_base_history = []
right_base_history = []

# Function to fit a polynomial or linear curve to the lane points and determine its degree

def fit_lane_curve(xs, ys):
    if len(xs) < 2:
        return None, 1
    linear = np.polyfit(ys, xs, 1)

    try:
        linear = np.polyfit(ys, xs, 1, rcond=1e-10)
        
        if len(xs) >= 3:
            quad = np.polyfit(ys, xs, 2, rcond=1e-10)
            
            y_vals = np.linspace(min(ys), max(ys), num=10)
            x_linear = np.polyval(linear, y_vals)
            x_quad = np.polyval(quad, y_vals)
            
            deviation = np.mean(np.abs(x_linear - x_quad))
            
            if deviation > 10:
                return quad, 2
    except Exception as e:
        print(f"Polyfit error: {str(e)}")
        return np.array([0.0, np.mean(xs)]), 1
        
    return linear, 1

# Function to fit lanes based on detected lines and previous lane curves

def fit_lanes(image, lines):
    global prev_left_curve, prev_left_degree, prev_right_curve, prev_right_degree
    height, width = image.shape[:2]

    left_points_x = []
    left_points_y = []
    right_points_x = []
    right_points_y = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            if abs(x2 - x1) < 1:
                continue
            
            slope = (y2 - y1) / (x2 - x1)
            mid_x = (x1 + x2) / 2

            if slope < -0.3 and mid_x < width * 0.5:
                left_points_x.extend([x1, x2])
                left_points_y.extend([y1, y2])
            elif slope > 0.2 and mid_x > width * 0.6:
                right_points_x.extend([x1, x2])
                right_points_y.extend([y1, y2])

    left_curve, left_degree = None, 1
    right_curve, right_degree = None, 1

    try:
        if len(left_points_x) >= 2:
            left_curve, left_degree = fit_lane_curve(left_points_x, left_points_y)
            if left_curve is not None:
                prev_left_curve, prev_left_degree = left_curve, left_degree
        elif prev_left_curve is not None:
            left_curve, left_degree = prev_left_curve, prev_left_degree
    except Exception as e:
        print(f"Failed to fit left lane curve: {str(e)}")
        if prev_left_curve is not None:
            left_curve, left_degree = prev_left_curve, prev_left_degree

    try:
        if len(right_points_x) >= 2:
            right_curve, right_degree = fit_lane_curve(right_points_x, right_points_y)
            if right_curve is not None:
                prev_right_curve, prev_right_degree = right_curve, right_degree
        elif prev_right_curve is not None:
            right_curve, right_degree = prev_right_curve, prev_right_degree
    except Exception as e:
        print(f"Failed to fit right lane curve: {str(e)}")
        if prev_right_curve is not None:
            right_curve, right_degree = prev_right_curve, prev_right_degree

    left_lane_points = []
    right_lane_points = []
    
    y_start = int(height * 0.65)
    y_end = height
    num_points = 25
    
    y_coords = np.linspace(y_start, y_end, num_points)
    
    if left_curve is not None:
        for y in y_coords:
            if left_degree == 2:
                x = left_curve[0] * y**2 + left_curve[1] * y + left_curve[2]
            else:
                x = left_curve[0] * y + left_curve[1]
            if 0 <= x < width:
                left_lane_points.append((int(x), int(y)))
    
    if right_curve is not None:
        for y in y_coords:
            if right_degree == 2:
                x = right_curve[0] * y**2 + right_curve[1] * y + right_curve[2]
            else:
                x = right_curve[0] * y + right_curve[1]
            if 0 <= x < width:
                right_lane_points.append((int(x), int(y)))
    
    return left_lane_points, right_lane_points


# Function to transform points back to the original image coordinates after perspective transformation

def transform_points_back(points, matrix):
    if not points:
        return []
    
    pts = np.array(points, dtype=np.float32)
    pts = pts.reshape(-1, 1, 2)
    
    transformed_pts = cv.perspectiveTransform(pts, matrix)
    
    return [tuple(map(int, pt[0])) for pt in transformed_pts]

# Function to find the base points of the lanes using histogram analysis with previous lane positions as fallback

def find_lane_base_points(warped_image, previous_left=None, previous_right=None):
    histogram = np.sum(warped_image[warped_image.shape[0]//2:,:], axis=0)
    
    car_position = warped_image.shape[1] // 2
    
    search_window = int(warped_image.shape[1] * 0.15)
    
    left_search_start = max(0, car_position - search_window * 2)
    left_search_end = car_position - int(search_window * 0.5)
    
    right_search_start = car_position + int(search_window * 0.5)
    right_search_end = min(warped_image.shape[1], car_position + search_window * 2)
    
    left_histogram = histogram[left_search_start:left_search_end]
    leftx_base = left_search_start + np.argmax(left_histogram) if np.max(left_histogram) > 0 else previous_left
    
    right_histogram = histogram[right_search_start:right_search_end]
    rightx_base = right_search_start + np.argmax(right_histogram) if np.max(right_histogram) > 0 else previous_right
    
    return leftx_base, rightx_base, histogram

# Function to track lanes using sliding windows approach

left_lane_counter = 0
right_lane_counter = 0

def track_lane_windows(binary, start_x, prev_x=None, max_empty=15, window_width=200, min_pix=40, max_pix=3500, max_step=10):
    height, width = binary.shape
    n_windows = 30
    window_height = height // n_windows
    current_x = start_x
    lane_points_x, lane_points_y = [], []
    empty_count = 0
    last_dx = 0

    for w in range(n_windows):
        y_top = height - (w+1)*window_height
        y_bot = height - w*window_height
        win_left = max(0, current_x - window_width//2)
        win_right = min(width, current_x + window_width//2)
        window = binary[y_top:y_bot, win_left:win_right]
        nonzero = np.nonzero(window)
        num_pix = len(nonzero[0])
        if min_pix < num_pix < max_pix:
            x_indices = nonzero[1] + win_left
            y_indices = nonzero[0] + y_top
            lane_points_x.extend(x_indices)
            lane_points_y.extend(y_indices)
            new_x = int(np.mean(x_indices))
            dx = new_x - current_x
            if abs(dx) > max_step:
                dx = np.sign(dx) * max_step
            last_dx = dx
            current_x = current_x + dx
            empty_count = 0
        else:
            empty_count += 1
            drift = abs(current_x - start_x)
            drift_thresh = width * 0.15
            extreme_drift = width * 0.4

            if drift > extreme_drift:
                histogram = np.sum(binary[y_top:y_bot, :], axis=0)
                if np.max(histogram) > 0:
                    current_x = int(np.argmax(histogram))
                    print(f"Extreme drift: re-initializing to global histogram peak at {current_x} (window {w})")
                empty_count = 0

            elif empty_count > max_empty and drift > drift_thresh:
                search_margin = int(width * 0.1)
                search_start = max(0, current_x - search_margin)
                search_end = min(width, current_x + search_margin)
                histogram = np.sum(binary[y_top:y_bot, search_start:search_end], axis=0)
                if np.max(histogram) > 0:
                    new_peak = search_start + int(np.argmax(histogram))
                    print(f"Re-initializing to local histogram peak at {new_peak} (window {w})")
                    current_x = new_peak
                empty_count = 0
            else:
                current_x += last_dx
    return lane_points_x, lane_points_y

def draw_center_line(img, left_points, right_points, thickness=5, color=(0, 0, 255)):
    center_line_img = np.zeros_like(img)
    if left_points and right_points and len(left_points) == len(right_points):
        for i in range(len(left_points)):
            lx, ly = left_points[i]
            rx, ry = right_points[i]
            cx, cy = int((lx+rx) / 2), int((ly+ry) / 2)

            if i > 0:
                prev_lx, prev_ly = left_points[i-1]
                prev_rx, prev_ry = right_points[i-1]
                prev_cx, prev_cy = int((prev_lx+prev_rx) / 2), int((prev_ly+prev_ry) / 2)
                cv.line(center_line_img, (prev_cx, prev_cy), (cx, cy), color, thickness)
    return center_line_img
# Function to create a lane path image based on detected lane points

def create_lane_path(image, left_points, right_points):
    lane_path = np.zeros_like(image)
    
    if not left_points or not right_points:
        return lane_path
    
    if left_points and right_points and len(left_points) > 1 and len(right_points) > 1:
        lane_full_points = np.vstack((left_points, right_points[::-1]))
        cv.fillPoly(lane_path, [np.array(lane_full_points, dtype=np.int32)], (0, 0, 255))  # Blue
    
    return lane_path

# Function to display lane lines on the original image

def display_lane_lines(image, left_points, right_points):
    line_image = np.zeros_like(image)
    
    if left_points and len(left_points) > 1:
        for i in range(len(left_points) - 1):
            pt1 = left_points[i]
            pt2 = left_points[i + 1]
            cv.line(line_image, pt1, pt2, (0, 255, 0), 8)  # Green
    
    if right_points and len(right_points) > 1:
        for i in range(len(right_points) - 1):
            pt1 = right_points[i]
            pt2 = right_points[i + 1]
            cv.line(line_image, pt1, pt2, (0, 0, 255), 8)  # Red
            
    return line_image

# Function to detect lanes in a single frame of video

def lane_detection(frame):
    global prev_left_curve, prev_left_degree, prev_right_curve, prev_right_degree
    global prev_left_base, prev_right_base, left_base_history, right_base_history
    global prev_left_points_x, prev_left_points_y, prev_right_points_x, prev_right_points_y

    if 'prev_left_points_x' not in globals():
        global prev_left_points_x, prev_left_points_y, prev_right_points_x, prev_right_points_y
        prev_left_points_x, prev_left_points_y = [], []
        prev_right_points_x, prev_right_points_y = [], []
        
    if 'prev_confidence' not in globals():
        global prev_confidence, confidence_history
        prev_confidence = 0
        confidence_history = []


    height, width = frame.shape[:2]
    
    roi_image, roi_polygon = roi(frame)
    warped_image, transform_matrix = perspective_transform(roi_image)
    binary = binary_conversion(warped_image)

    white_ratio = np.sum(binary == 255) / binary.size
    print("White ratio:", int(white_ratio * 100), "%")
    if white_ratio > 0.98 or white_ratio < 0.005:
        print("Binary image saturated, resetting lane history.")
        prev_left_points_x, prev_left_points_y = [], []
        prev_right_points_x, prev_right_points_y = [], []
        left_base_history.clear()
        right_base_history.clear()
        prev_left_base = None
        prev_right_base = None

    left_base_x, right_base_x, histogram = find_lane_base_points(binary, prev_left_base, prev_right_base)

    if left_base_x is None:
        left_base_x = width // 4
    if right_base_x is None:
        right_base_x = width * 3 // 4

    if prev_left_base is not None and abs(left_base_x - prev_left_base) > binary.shape[1] * 0.05:
        left_base_x = prev_left_base
    
    if prev_right_base is not None and abs(right_base_x - prev_right_base) > binary.shape[1] * 0.05:
        right_base_x = prev_right_base
    
    left_base_history.append(left_base_x)
    right_base_history.append(right_base_x)
    
    if len(left_base_history) > lane_history_size:
        left_base_history.pop(0)
    if len(right_base_history) > lane_history_size:
        right_base_history.pop(0)
    
    left_base_x = int(np.mean(left_base_history))
    right_base_x = int(np.mean(right_base_history))
    
    prev_left_base = left_base_x
    prev_right_base = right_base_x

    left_points_x, left_points_y = track_lane_windows(
        binary, left_base_x, prev_left_base
        )
    
    right_points_x, right_points_y = track_lane_windows(
        binary, right_base_x, prev_right_base
        )
    
    
    left_curve, left_degree = fit_lane_curve(left_points_x, left_points_y) if len(left_points_x) >= 2 else (prev_left_curve, prev_left_degree)
    right_curve, right_degree = fit_lane_curve(right_points_x, right_points_y) if len(right_points_x) >= 2 else (prev_right_curve, prev_right_degree)
    
    if left_curve is not None:
        prev_left_curve, prev_left_degree = left_curve, left_degree
    if right_curve is not None:
        prev_right_curve, prev_right_degree = right_curve, right_degree

    
    left_points, right_points = [], []
    y_start = int(height * 0.15)
    y_end = height
    num_points = 45
    y_coords = np.linspace(y_start, y_end, num_points)
    
    if left_curve is not None:
        for y in y_coords:
            if left_degree == 2:
                x = left_curve[0] * y**2 + left_curve[1] * y + left_curve[2]
            else:
                x = left_curve[0] * y + left_curve[1]
            if 0 <= x < width:
                left_points.append((int(x), int(y)))
    
    if right_curve is not None:
        for y in y_coords:
            if right_degree == 2:
                x = right_curve[0] * y**2 + right_curve[1] * y + right_curve[2]
            else:
                x = right_curve[0] * y + right_curve[1]
            if 0 <= x < width:
                right_points.append((int(x), int(y)))
    
    inv_transform_matrix = np.linalg.inv(transform_matrix)
    left_points_original = transform_points_back(left_points, inv_transform_matrix)
    right_points_original = transform_points_back(right_points, inv_transform_matrix)
    
    print(f"Lines detected: {0 if len(left_points_x) + len(right_points_x) == 0 else len(left_points_x) + len(right_points_x)}")
    
    cv.imshow("Binary Image", cv.resize(binary, (400, 300)))
    
    plt_histogram = np.zeros((400, binary.shape[1], 3), dtype=np.uint8)
    if np.max(histogram) > 0:
        hist_normalized = histogram/np.max(histogram)*300
        for i in range(binary.shape[1]):
            if hist_normalized[i] > 0:
                cv.line(plt_histogram, (i, 399), (i, 399-int(hist_normalized[i])), (0, 0, 255), 1)
    cv.line(plt_histogram, (left_base_x, 0), (left_base_x, 399), (0, 255, 0), 2)
    cv.line(plt_histogram, (right_base_x, 0), (right_base_x, 399), (0, 0, 255), 2)

    hist_resized = cv.resize(plt_histogram, (400, 300))
    cv.imshow("Histogram", hist_resized)
    
    point_debug = warped_image.copy()
    
    for x, y in zip(left_points_x, left_points_y):
        x_int, y_int = int(x), int(y)
        if 0 <= x_int < point_debug.shape[1] and 0 <= y_int < point_debug.shape[0]:
            cv.circle(point_debug, (x_int, y_int), 3, (0, 255, 0), -1)
    
    for x, y in zip(right_points_x, right_points_y):
        x_int, y_int = int(x), int(y)
        if 0 <= x_int < point_debug.shape[1] and 0 <= y_int < point_debug.shape[0]:
            cv.circle(point_debug, (x_int, y_int), 3, (0, 0, 255), -1)
    
    window_height = binary.shape[0] // 9
    for window in range(9):
        y_top = binary.shape[0] - (window+1) * window_height
        y_bottom = binary.shape[0] - window * window_height
        
        left_x = left_base_x
        if len(left_points_y) > 0:
            window_indices = [i for i, y_val in enumerate(left_points_y) 
                            if y_top <= y_val < y_bottom]
            if window_indices:
                left_x = int(np.mean([left_points_x[i] for i in window_indices]))
        
        cv.rectangle(point_debug, 
                    (int(left_x - 30), y_top), 
                    (int(left_x + 30), y_bottom),
                    (0, 255, 255), 2)
        
        right_x = right_base_x
        if len(right_points_y) > 0:
            window_indices = [i for i, y_val in enumerate(right_points_y) 
                            if y_top <= y_val < y_bottom]
            if window_indices:
                right_x = int(np.mean([right_points_x[i] for i in window_indices]))
        
        cv.rectangle(point_debug, 
                    (int(right_x - 30), y_top), 
                    (int(right_x + 30), y_bottom),
                    (0, 255, 255), 2)
    
    cv.imshow("Detected Points", cv.resize(point_debug, (400, 300)))
    
    return left_points_original, right_points_original, roi_image, roi_polygon, warped_image
    
# Function to process the video and apply lane detection on each frame

def process_video(video_path):
    cap = cv.VideoCapture(video_path)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    
    while cap.isOpened():
        if 'seek_to' in locals():
            cap.set(cv.CAP_PROP_POS_FRAMES, frame_count)
            del seek_to
        ret, frame = cap.read()
        if not ret:
            break
            
        left_points, right_points, cropped_image, roi_polygon, warped_image = lane_detection(frame)
        
        lane_lines = display_lane_lines(frame, left_points, right_points)
        lane_path = create_lane_path(frame, left_points, right_points)

        center_line = draw_center_line(frame, left_points, right_points)
        
        combo_image = cv.addWeighted(frame, 0.9, lane_lines, 1, 1)
        combo_image = cv.addWeighted(combo_image, 0.8, lane_path, 0.3, 0)
        combo_image = cv.addWeighted(combo_image, 1.0, center_line, 1.0, 0)
        
        cv.imshow('Lane Detection', combo_image)
        
        # roi_border_img = cropped_image.copy()
        # roi_polygon_int = roi_polygon.astype(np.int32)
        # cv.polylines(roi_border_img, [roi_polygon_int], isClosed=True, color=(255, 0, 255), thickness=3)
        # roi_resized = cv.resize(roi_border_img, (400, 300))
        # cv.imshow("ROI-Image", roi_resized)
        
        # warped_resized = cv.resize(warped_image, (400, 300))
        # cv.imshow("Bird's Eye View", warped_resized)
        
        
        key = cv.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('l'):  # Skip forward 30 frames
            frame_count = min(frame_count + 30, total_frames - 1)
            seek_to = True
            continue
        elif key == ord('j'):  # Skip backward 30 frames
            frame_count = max(frame_count - 30, 0)
            seek_to = True
            continue
        else:
            frame_count += 1
            
    cap.release()
    cv.destroyAllWindows()

# Main function to run the lane detection on a video file
if __name__ == "__main__":
    video_path = "../test_vids/nl_highway.mp4"
    print(f"Processing video: {video_path}")
    process_video(video_path)
