import numpy as np

from src.perception.lane_detection.cv.perspective import get_roi_range, bev_m_per_px, bev_row_for_distance


LANE_WIDTH_M = 3.55
MAX_OFFSET_JUMP_M = 0.5  # max change of deviation between two frames before we stop trusting it


def validate_lane_geometry(left_fitx, right_fitx, ploty):
    """
    Validate lane geometry for reasonable lane width and positioning.
    
    Returns:
        bool: True if geometry is valid, False otherwise
        lane_width_pix: Lane width in pixels (if valid)
    """

    if len(left_fitx) == 0 or len(right_fitx) == 0:
        print("Empty lane fit arrays detected")
        return False, None

    leftx = left_fitx[::-1]
    rightx = right_fitx[::-1]

    left_bottom = leftx[-1]
    right_bottom = rightx[-1]
    lane_width_pix = right_bottom - left_bottom

    # Check for crossed lanes (removed - allow lanes to converge)
    if left_bottom > right_bottom:
        print(f"Lane lines reversed: left={left_bottom:.1f}, right={right_bottom:.1f}")
        return False, None

    return True, lane_width_pix


def check_lane_width_outliers(lane_width_pix):
    """
    Check if current lane width is an outlier based on historical data.
    
    Returns:
        bool: True if lane width is acceptable, False if it's an outlier
    """

    if not hasattr(check_lane_width_outliers, 'lane_width_history'):
        check_lane_width_outliers.lane_width_history = []
    if not hasattr(check_lane_width_outliers, 'outlier_count'):
        check_lane_width_outliers.outlier_count = 0
    if not hasattr(check_lane_width_outliers, 'consecutive_outliers'):
        check_lane_width_outliers.consecutive_outliers = 0
        
    lane_width_history = check_lane_width_outliers.lane_width_history

    lane_width_history.append(lane_width_pix)
    if len(lane_width_history) > 30:
        lane_width_history.pop(0)
        
    if len(lane_width_history) >= 15:
        avg_lane_width = np.mean(lane_width_history)
        # Tighter tolerance: only allow 10% variation instead of 15%
        max_lane_width = avg_lane_width * 1.10
        min_lane_width = avg_lane_width * 0.90
        if lane_width_pix < min_lane_width or lane_width_pix > max_lane_width:
            check_lane_width_outliers.outlier_count += 1
            print(f"Lane width outlier: {lane_width_pix:.1f} pixels (avg={avg_lane_width:.1f}), grace count: {check_lane_width_outliers.outlier_count}")
            
            check_lane_width_outliers.consecutive_outliers += 1
            
            if check_lane_width_outliers.outlier_count >= 2:
                check_lane_width_outliers.outlier_count = 0
                
                if check_lane_width_outliers.consecutive_outliers >= 7:
                    print("Resetting lane width history due to outliers")
                    lane_width_history.clear()
                    check_lane_width_outliers.consecutive_outliers = 0
                    
                return False
        else:
            check_lane_width_outliers.outlier_count = 0
            check_lane_width_outliers.consecutive_outliers = 0
    else:
        if lane_width_pix > 700:
            print(f"Unreasonably wide lane width: {lane_width_pix:.1f} pixels")
            return False

    return True


def check_offset_jump(deviation_m, max_jump=MAX_OFFSET_JUMP_M, max_rejects=3):
    """
    Reject frames where the deviation jumps more than max_jump since the last accepted frame
    (typically the detector switching to a neighbour lane for one frame).
    After max_rejects rejections in a row the new value is accepted, so a real change is not ignored forever.

    Returns:
        bool: True if deviation is acceptable, False if it's a jump
    """

    if not hasattr(check_offset_jump, 'previous_deviation'):
        check_offset_jump.previous_deviation = None
        check_offset_jump.rejects = 0

    previous = check_offset_jump.previous_deviation
    if previous is not None and abs(deviation_m - previous) > max_jump and check_offset_jump.rejects < max_rejects:
        check_offset_jump.rejects += 1
        print(f"Deviation jump rejected: {previous:.2f}m -> {deviation_m:.2f}m ({check_offset_jump.rejects}/{max_rejects})")
        return False

    check_offset_jump.previous_deviation = deviation_m
    check_offset_jump.rejects = 0
    return True


def curvature_radius(fit, y_eval, mx_per_px, my_per_px):
    """
    Radius of curvature in metres of x = a*y^2 + b*y + c (fit in BEV pixels) at row y_eval.
    Returns 1e6 for a straight line.
    """
    a_px, b_px, _ = fit
    a_m = mx_per_px / (my_per_px ** 2) * a_px
    b_m = mx_per_px / my_per_px * b_px
    y_m = y_eval * my_per_px
    if abs(a_m) < 1e-7:
        return 1e6
    return (1 + (2 * a_m * y_m + b_m) ** 2) ** 1.5 / abs(2 * a_m)


def calculate_curvature_and_deviation(left_fit, right_fit, left_fitx, right_fitx, ploty, bev_shape, speed=0, lookahead_m=None, lane_width_m=LANE_WIDTH_M):
    """
    Calculate lane curvature and vehicle deviation from lane center, at the bottom of the BEV and at a lookahead distance.

    Sign: deviation > 0 means the vehicle is right of the lane center.

    Args:
        left_fit, right_fit: 2nd order polynomial fits x(y) of the lane lines in BEV pixels
        left_fitx, right_fitx, ploty: Evaluated lane lines (ploty descending, index 0 = bottom row)
        bev_shape: Shape of the warped image
        speed: Vehicle speed in kph (the BEV region depends on it)
        lookahead_m: Distance from the camera to measure deviation_ahead at, clipped to the BEV region.
                     None means the bottom row.
        lane_width_m: Real lane width used for the lateral scale
    Returns:
        dict with deviation, deviation_ahead, lookahead_m, curvature_radius, lane_center, vehicle_center, lane_width
        or None if validation fails
    """

    if left_fit is None or right_fit is None:
        return None

    # Validate lane geometry
    is_valid, lane_width_pix = validate_lane_geometry(left_fitx, right_fitx, ploty)
    if not is_valid:
        return None

    # Check for lane width outliers
    if not check_lane_width_outliers(lane_width_pix):
        return None

    # Lateral scale from the detected lane width, longitudinal scale from the warp geometry
    xm_per_pix = lane_width_m / lane_width_pix
    _, ym_per_pix = bev_m_per_px(bev_shape, speed)

    if xm_per_pix <= 0 or xm_per_pix > 0.1:
        print(f"Unreasonable xm_per_pix: {xm_per_pix:.4f}")
        return None

    h, w = bev_shape[0], bev_shape[1]
    vehicle_center = w / 2.0

    # Deviation at the bottom row (closest to the car)
    y_bottom = h - 1
    lane_center = (np.polyval(left_fit, y_bottom) + np.polyval(right_fit, y_bottom)) / 2.0
    deviation_m = (vehicle_center - lane_center) * xm_per_pix

    # Detector jumped onto a neighbour lane
    if abs(deviation_m) > lane_width_m / 2:
        print(f"Deviation larger than half a lane: {deviation_m:.2f}m")
        return None
    if not check_offset_jump(deviation_m):
        return None

    # Deviation at the lookahead distance, kept inside the warped region
    near, far = get_roi_range(speed)
    if lookahead_m is None:
        lookahead_m = near
    lookahead_m = float(np.clip(lookahead_m, near, far - 1.0))
    y_ahead = bev_row_for_distance(lookahead_m, bev_shape, speed)
    lane_center_ahead = (np.polyval(left_fit, y_ahead) + np.polyval(right_fit, y_ahead)) / 2.0
    deviation_ahead_m = (vehicle_center - lane_center_ahead) * xm_per_pix

    left_curverad = curvature_radius(left_fit, y_ahead, xm_per_pix, ym_per_pix)
    right_curverad = curvature_radius(right_fit, y_ahead, xm_per_pix, ym_per_pix)
    curverad = (left_curverad + right_curverad) / 2.0

    return {
        'deviation': deviation_m,
        'deviation_ahead': deviation_ahead_m,
        'lookahead_m': lookahead_m,
        'curvature_radius': curverad,
        'lane_center': lane_center,
        'vehicle_center': vehicle_center,
        'lane_width': lane_width_pix,
    }
