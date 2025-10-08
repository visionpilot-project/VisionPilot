import numpy as np
import cv2

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

def perspective_warp(img, speed=0, debugger=None):
    img_size = (img.shape[1], img.shape[0])
    w, h = img_size

    src = get_src_points(img.shape, speed)

    dst = np.float32([
        [w*0.2, h],
        [w*0.8, h],
        [w*0.8, 0],
        [w*0.2, 0]
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    binary_warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    # Debug perspective transform
    if debugger:
        debugger.debug_perspective_transform(img, binary_warped, src, dst)
    
    return binary_warped, Minv


def debug_perspective_live(img, speed_kph):
    debug_img = img.copy()
    
    src_points = get_src_points(img.shape, speed_kph)
    
    src_int = src_points.astype(np.int32)
    
    cv2.polylines(debug_img, [src_int], isClosed=True, color=(0, 255, 0), thickness=2)
    
    labels = ['Bottom Left', 'Bottom Right', 'Top Right', 'Top Left']
    colors = [(0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255)]
    
    for i, (point, label, color) in enumerate(zip(src_int, labels, colors)):
        cv2.circle(debug_img, tuple(point), 5, color, -1)
        cv2.putText(debug_img, label, (point[0] + 10, point[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    cv2.putText(debug_img, f"Speed: {speed_kph:.1f} km/h", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(debug_img, "Perspective Transform Region", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(debug_img, "Green: Transform Area", (10, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imshow('Perspective Transform Debug', debug_img)

def debug_perspective_points(img, speeds=[0, 60, 120]):
    img_size = (img.shape[1], img.shape[0])
    w, h = img_size
    left_bottom  = [118, 608]
    right_bottom = [1077, 596]
    top_right    = [730, 408]
    top_left     = [519, 408]

    baseline_points = np.array([left_bottom, right_bottom, top_right, top_left], np.int32)
    img_base = img.copy()
    cv2.polylines(img_base, [baseline_points], isClosed=True, color=(0,255,0), thickness=2)
    cv2.imshow("Baseline Src Points", img_base)

    for speed in speeds:
        speed_norm = min(speed / 120.0, 1.0)
        y_shift_norm = min(speed_norm, 0.5)
        top_shift = -40 * y_shift_norm
        side_shift_norm = min(speed_norm, 0.5)
        side_shift = 100 * side_shift_norm
        shifted_points = np.array([
            left_bottom,
            right_bottom,
            [top_right[0] - side_shift, int(top_right[1] + top_shift)],
            [top_left[0] + side_shift,  int(top_left[1]  + top_shift)]
        ], np.int32)
        img_speed = img.copy()
        cv2.polylines(img_speed, [shifted_points], isClosed=True, color=(0,0,255), thickness=2)
        cv2.putText(img_speed, f"Speed: {speed} km/h", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow(f"Src Points at {speed} km/h", img_speed)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    img = cv2.imread("/Users/jstamm2024/Documents/GitHub/self-driving-car-simulation/images/beamng-bonnet.png")
    debug_perspective_points(img)
