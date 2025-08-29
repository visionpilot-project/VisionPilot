import numpy as np


def calculate_curvature_and_deviation(ploty, left_fitx, right_fitx, binary_warped, debugger=None):
    if not hasattr(calculate_curvature_and_deviation, 'lane_width_history'):
        calculate_curvature_and_deviation.lane_width_history = []
    lane_width_history = calculate_curvature_and_deviation.lane_width_history

    ym_per_pix = 30/720  # meters per pixel in y dimension

    # Check for empty arrays first
    if len(left_fitx) == 0 or len(right_fitx) == 0:
        print("Empty lane fit arrays detected")
        return None, None, None, None, None

    leftx = left_fitx[::-1]
    rightx = right_fitx[::-1]

    y_eval = np.max(ploty)

    left_bottom = leftx[-1]
    right_bottom = rightx[-1]
    lane_width_pix = right_bottom - left_bottom

    lane_width_history.append(lane_width_pix)
    if len(lane_width_history) > 30:
        lane_width_history.pop(0)

    if not hasattr(calculate_curvature_and_deviation, 'outlier_count'):
        calculate_curvature_and_deviation.outlier_count = 0
    if not hasattr(calculate_curvature_and_deviation, 'consecutive_outliers'):
        calculate_curvature_and_deviation.consecutive_outliers = 0
        
    if len(lane_width_history) >= 15:
        avg_lane_width = np.mean(lane_width_history)
        max_lane_width = avg_lane_width * 1.15
        min_lane_width = avg_lane_width * 0.85
        if lane_width_pix < min_lane_width or lane_width_pix > max_lane_width:
            calculate_curvature_and_deviation.outlier_count += 1
            print(f"Lane width outlier: {lane_width_pix:.1f} pixels (avg={avg_lane_width:.1f}), grace count: {calculate_curvature_and_deviation.outlier_count}")
            
            # Increment consecutive outliers counter
            calculate_curvature_and_deviation.consecutive_outliers += 1
            
            if calculate_curvature_and_deviation.outlier_count >= 2:
                calculate_curvature_and_deviation.outlier_count = 0
                
                # Reset lane width history if we've seen too many consecutive outliers
                if calculate_curvature_and_deviation.consecutive_outliers >= 10:
                    print("Resetting lane width history due to persistent outliers")
                    lane_width_history.clear()
                    calculate_curvature_and_deviation.consecutive_outliers = 0
                    
                return None, None, None, None, None
        else:
            calculate_curvature_and_deviation.outlier_count = 0
            calculate_curvature_and_deviation.consecutive_outliers = 0
    else:
        if lane_width_pix < 50 or lane_width_pix > 700:
            print(f"Unreasonable lane width: {lane_width_pix:.1f} pixels")
            return None, None, None, None, None

    if left_bottom >= right_bottom:
        print(f"Lane lines crossed: left={left_bottom:.1f}, right={right_bottom:.1f}")
        return None, None, None, None, None

    xm_per_pix = 3.55 / lane_width_pix

    if xm_per_pix <= 0 or xm_per_pix > 0.1:
        print(f"Unreasonable xm_per_pix: {xm_per_pix:.4f}")
        return None, None, None, None, None

    try:
        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) \
                        / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) \
                         / np.absolute(2*right_fit_cr[0])

        max_reasonable_curve = 1000  # meters
        if left_curverad > max_reasonable_curve or right_curverad > max_reasonable_curve:
            left_curverad = min(left_curverad, max_reasonable_curve)
            right_curverad = min(right_curverad, max_reasonable_curve)

        lane_center = (left_bottom + right_bottom) / 2.0
        vehicle_center = binary_warped.shape[1] / 2.0
        deviation_m = (vehicle_center - lane_center) * xm_per_pix

        max_reasonable_deviation = 0.75  # meters
        if abs(deviation_m) > max_reasonable_deviation:
            print(f"Unreasonable deviation detected: {deviation_m:.2f}m, clipping to {max_reasonable_deviation:.2f}m")
            deviation_m = np.clip(deviation_m, -max_reasonable_deviation, max_reasonable_deviation)

        if debugger:
            debugger.debug_metrics(left_curverad, right_curverad, deviation_m, lane_center, 
                                 vehicle_center, lane_width_pix, xm_per_pix)

        return left_curverad, right_curverad, deviation_m, lane_center, vehicle_center
        
    except Exception as e:
        print(f"Error in curvature calculation: {e}")
        return None, None, None, None, None
