import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from beamng_sim.lane_detection.cv.thresholding import apply_thresholds
from beamng_sim.lane_detection.perspective import debug_perspective_live, get_src_points, perspective_warp
from beamng_sim.lane_detection.cv.lane_finder import get_histogram, sliding_window_search
from beamng_sim.lane_detection.metrics import calculate_curvature_and_deviation, process_deviation
from beamng_sim.lane_detection.visualization import draw_lane_overlay, add_text_overlay, create_mask_overlay
from beamng_sim.lane_detection.unet.postprocess import process_unet_mask, run_unet_on_frame

from beamng_sim.lane_detection.confidence import compute_confidence_cv
from beamng_sim.lane_detection.confidence import compute_confidence_unet


import numpy as np
import cv2


def process_frame_cv(img, speed=0, previous_steering=0, debug_display=False, perspective_debug_display=False):
        
    previous_fit = None
    confidence = 0.0
    try:

        src_points = get_src_points(img.shape, speed, previous_steering)

        binary_image, avg_brightness = apply_thresholds(img, src_points=src_points, debug_display=debug_display)
        if debug_display:
            cv2.imshow('1. Binary Image CV', binary_image*255 if binary_image.max()<=1 else binary_image)
            cv2.waitKey(1)
        
        binary_warped, Minv = perspective_warp(binary_image, speed=speed)
        if perspective_debug_display:
            debug_perspective_live(img, speed, previous_steering=0)
        
        if debug_display:
            warped_display = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
            cv2.imshow('2. Warped Binary CV', warped_display)
        
        histogram = get_histogram(binary_warped)
        ploty, left_fit, right_fit, left_fitx, right_fitx = sliding_window_search(binary_warped, histogram)

        current_fit = (left_fit, right_fit)
        
        if debug_display:
            lane_img = np.zeros_like(img)
            
            if len(left_fitx) > 0 and len(right_fitx) > 0 and len(ploty) > 0:
                try:
                    left_points = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
                    cv2.polylines(lane_img, np.int32([left_points]), False, (255, 0, 0), 8)
                    
                    right_points = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
                    cv2.polylines(lane_img, np.int32([right_points]), False, (0, 0, 255), 8)
                    
                except Exception as lane_err:
                    print(f"Debug visualization error CV: {lane_err}")
            else:
                print("Insufficient lane pixels detected CV Method!")

            cv2.imshow('3. Lane Lines CV', lane_img)

        metrics_result = calculate_curvature_and_deviation(ploty, left_fitx, right_fitx, binary_warped)

        confidence = compute_confidence_cv(left_fitx, right_fitx, ploty, current_fit=current_fit, previous_fit=previous_fit)

        previous_fit = current_fit

        if metrics_result is None or (isinstance(metrics_result, tuple) and all(x is None for x in metrics_result)):
            left_curverad, right_curverad, deviation, lane_center, vehicle_center, lane_width = None, None, None, None, None, None
            smoothed_deviation = 0.0
            effective_deviation = 0.0
            print("Lane detection metrics calculation returned None values")
        else:
            if len(metrics_result) == 6:
                left_curverad, right_curverad, deviation, lane_center, vehicle_center, lane_width = metrics_result
            elif len(metrics_result) == 5:
                left_curverad, right_curverad, deviation, lane_center, vehicle_center = metrics_result
                lane_width = None
            else:
                left_curverad, right_curverad, deviation, lane_center, vehicle_center, lane_width = None, None, None, None, None, None
            smoothed_deviation, effective_deviation = process_deviation(deviation)

        result = draw_lane_overlay(img, binary_warped, Minv, left_fitx, right_fitx, ploty, deviation)
        result = add_text_overlay(result, left_curverad, right_curverad, deviation, avg_brightness, speed, confidence=confidence)
        
        metrics = {
            'left_curverad': left_curverad,
            'right_curverad': right_curverad,
            'deviation': deviation,
            'smoothed_deviation': smoothed_deviation,
            'effective_deviation': effective_deviation,
            'lane_center': lane_center,
            'vehicle_center': vehicle_center,
            'lane_width': lane_width,
            'confidence': confidence
        }
        
        return result, metrics, confidence
        
    except Exception as e:
        print(f"Lane detection error CV: {e}")
        result = img.copy()
        metrics = {
            'left_curverad': 0,
            'right_curverad': 0,
            'deviation': 0,
            'smoothed_deviation': 0,
            'effective_deviation': 0,
            'lane_center': 0,
            'vehicle_center': 0,
            'lane_width': 0,
            'confidence': 0,
            'error': str(e)
        }
        return result, metrics, confidence
    
def process_frame_unet(img, model, speed=0, previous_steering=0, debug_display=False):
        
    previous_fit = None
    confidence = 0.0
    try:
        raw_mask = run_unet_on_frame(img, model)
        
        if debug_display:
            display_raw_mask = cv2.resize(raw_mask * 255, (img.shape[1], img.shape[0]))
            cv2.imshow('1. Raw UNet Prediction', display_raw_mask)
        
        src_points = get_src_points(img.shape, speed, previous_steering)
        
        scale_x = raw_mask.shape[1] / img.shape[1]  # 320 / original_width
        scale_y = raw_mask.shape[0] / img.shape[0]  # 256 / original_height
        
        scaled_src_points = src_points.copy()
        scaled_src_points[:, 0] *= scale_x
        scaled_src_points[:, 1] *= scale_y
        
        mask = process_unet_mask(raw_mask, scaled_src_points, min_area=100)
        
        if debug_display:
            display_mask = cv2.resize(mask * 255, (img.shape[1], img.shape[0]))
            cv2.imshow('2. Processed UNet Mask', display_mask)
        
        debug_mask_with_points = cv2.cvtColor(mask.copy()*255, cv2.COLOR_GRAY2BGR)
        for point in scaled_src_points:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:  # Check bounds
                cv2.circle(debug_mask_with_points, (x, y), 5, (0, 0, 255), -1)  # Red circle
        
        if debug_display:
            cv2.imshow('Debug: Mask with Scaled Points', debug_mask_with_points)
        resized_mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        
        binary_warped, Minv = perspective_warp(resized_mask, speed=speed)
        
        if debug_display:
            warped_display = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
            cv2.imshow('3. Warped Binary UNet', warped_display)

        histogram = get_histogram(binary_warped)
        ploty, left_fit, right_fit, left_fitx, right_fitx = sliding_window_search(binary_warped, histogram)


        current_fit = (left_fit, right_fit)
        
        if debug_display:
            lane_img = np.zeros_like(img)
            
            if len(left_fitx) > 0 and len(right_fitx) > 0 and len(ploty) > 0:
                try:
                    left_points = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
                    cv2.polylines(lane_img, np.int32([left_points]), False, (255, 0, 0), 8)
                    
                    right_points = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
                    cv2.polylines(lane_img, np.int32([right_points]), False, (0, 0, 255), 8)
                    
                except Exception as lane_err:
                    print(f"Debug visualization error UNet: {lane_err}")
            else:
                print("Insufficient lane pixels detected UNet Method!")

            cv2.imshow('4. Lane Lines UNet', lane_img)

        metrics_result = calculate_curvature_and_deviation(ploty, left_fitx, right_fitx, binary_warped)
        confidence = compute_confidence_unet(left_fitx, right_fitx, ploty, current_fit=current_fit, previous_fit=previous_fit)

        previous_fit = current_fit

        if metrics_result == (None, None, None, None, None, None):
            left_curverad, right_curverad, deviation, lane_center, vehicle_center, lane_width = None, None, None, None, None, None
            smoothed_deviation = 0.0
            effective_deviation = 0.0
            print("Lane detection metrics calculation returned None values")
        else:
            left_curverad, right_curverad, deviation, lane_center, vehicle_center, lane_width = metrics_result
            smoothed_deviation, effective_deviation = process_deviation(deviation)

        result = draw_lane_overlay(img.copy(), binary_warped, Minv, left_fitx, right_fitx, ploty, deviation)
        
        result = create_mask_overlay(result, mask, alpha=0.3, color=(0, 0, 255))
        
        result = add_text_overlay(result, left_curverad, right_curverad, deviation, 0, speed, confidence=confidence)
        
        metrics = {
            'left_curverad': left_curverad,
            'right_curverad': right_curverad,
            'deviation': deviation,
            'smoothed_deviation': smoothed_deviation,
            'effective_deviation': effective_deviation,
            'lane_center': lane_center,
            'vehicle_center': vehicle_center,
            'lane_width': lane_width,
            'confidence': confidence
        }

        return result, metrics, confidence

    except Exception as e:
        print(f"Lane detection error UNET: {e}")
        result = img.copy()
        metrics = {
            'left_curverad': 0,
            'right_curverad': 0,
            'deviation': 0,
            'smoothed_deviation': 0,
            'effective_deviation': 0,
            'lane_center': 0,
            'vehicle_center': 0,
            'lane_width': 0,
            'confidence': 0,
            'error': str(e)
        }
        return result, metrics, confidence
