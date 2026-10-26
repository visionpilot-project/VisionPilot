import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.perception.lane_detection.cv.thresholding import apply_thresholds_with_voting
from src.perception.lane_detection.cv.perspective import debug_perspective_live, get_src_points, perspective_warp
from src.perception.lane_detection.cv.lane_finder import get_histogram, detect_lane_type, fill_dashed_lane_gaps
from src.perception.lane_detection.metrics import calculate_curvature_and_deviation
from src.perception.lane_detection.visualization import draw_multiple_lanes_overlay, add_text_overlay

from src.perception.lane_detection.cv.multi_lane.multi_lane_finder import detect_multiple_lanes
from src.perception.lane_detection.cv.multi_lane.lane_selector import get_current_lane
from src.perception.lane_detection.ufld.main import UFLDv2Inference
from config.config import MODELS_DIR

import numpy as np
import cv2
from pathlib import Path


_ufld_model = None


def _get_ufld_model():
    """Lazy-load UFLDv2 model on first use."""
    global _ufld_model
    if _ufld_model is None:
        model_path = MODELS_DIR / "ufld" / "ufld_culane_res18.pth"
        config_path = Path(__file__).parent.parent.parent.parent / "ufldv2" / "configs" / "ufld_culane_res18.py"

        try:
            _ufld_model = UFLDv2Inference(str(model_path), str(config_path))
            print("[MAIN] UFLDv2 model loaded successfully")
        except Exception as e:
            print(f"[MAIN] Failed to load UFLDv2 model: {e}")
            _ufld_model = False

    return _ufld_model if _ufld_model is not False else None


def process_frame_ufld(img):
    """Run UFLD once and return its lane mask for later CV processing."""
    ufld_model = _get_ufld_model()
    if ufld_model is None:
        return None
    try:
        return ufld_model.infer(img)
    except Exception as e:
        print(f"[MAIN] UFLDv2 inference failed: {e}")
        return None


def process_frame_cv(img, speed=0, previous_steering=0, debug_display=False, perspective_debug_display=False, calibration_data=None, vehicle_model='q8_andronisk', num_lanes=3, lookahead_m=None, ufld_mask=None):

    try:

        src_points = get_src_points(img.shape, speed, previous_steering, vehicle_model=vehicle_model, calibration_data=calibration_data)

        # Apply thresholding to full image with UFLD as voting feature
        binary_image, avg_brightness = apply_thresholds_with_voting(
            img,
            src_points=None,
            debug_display=debug_display,
            ufld_mask=ufld_mask
        )

        if debug_display:
            binary_display_resized = cv2.resize(binary_image, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow('Binary Image', binary_display_resized)

        # Apply roi mask
        mask = np.zeros(binary_image.shape[:2], dtype=np.uint8)
        src_poly = np.array(src_points, dtype=np.int32)
        cv2.fillPoly(mask, [src_poly], 1)
        binary_image = binary_image * mask
        
        # perspective warp for BEV
        binary_warped, Minv = perspective_warp(binary_image, speed=speed, calibration_data=calibration_data, vehicle_model=vehicle_model)

        is_dashed_lane = detect_lane_type(binary_warped)
        if is_dashed_lane:
            binary_warped = fill_dashed_lane_gaps(binary_warped, gap_size=20)
        
        if perspective_debug_display:
            debug_perspective_live(img, speed, previous_steering=0, vehicle_model=vehicle_model, calibration_data=calibration_data)
        
        if debug_display:
            warped_display = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
            warped_display_resized = cv2.resize(warped_display, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow('Warped Binary CV', warped_display_resized)

        lanes = None
        detected_num_lanes = num_lanes

        for attempt_lanes in [num_lanes, num_lanes-1, 2, 1]:
            if attempt_lanes < 1:
                break
            # SET TO FALSE TEMPORARILY
            lanes = detect_multiple_lanes(binary_warped, num_lanes=attempt_lanes, debug=False)
            if lanes is not None:
                detected_num_lanes = attempt_lanes
                print(f"[MULTI_LANE] Successfully detected {detected_num_lanes} lanes")
                break
            else:
                print(f"[MULTI_LANE] failed to detect {attempt_lanes} lanes, trying with {attempt_lanes-1} lanes")
        

        if lanes is None:
            print("[Multi-Lane] detection failed")
            result = img.copy()
            metrics = {
                'valid': False,
                'current_lane': None,
                'all_lanes': None,
                'error': 'Multi-lane detection failed'
            }
            return result, metrics
        
        # show lanes in warped space
        if debug_display:
            warped_lane_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            
            for lane_idx, lane in enumerate(lanes):
                ploty_viz = lane['ploty'].astype(np.int32)
                left_fitx_viz = lane['left_fitx'].astype(np.int32)
                right_fitx_viz = lane['right_fitx'].astype(np.int32)
                color = colors[lane_idx % len(colors)]
                
                # clip
                h, w = binary_warped.shape[:2]
                left_fitx_viz = np.clip(left_fitx_viz, 0, w - 1)
                right_fitx_viz = np.clip(right_fitx_viz, 0, w - 1)
                ploty_viz = np.clip(ploty_viz, 0, h - 1)
                
                if len(left_fitx_viz) > 0 and len(ploty_viz) > 0:
                    left_points = np.array([np.transpose(np.vstack([left_fitx_viz, ploty_viz]))], dtype=np.int32)
                    cv2.polylines(warped_lane_img, left_points, isClosed=False, color=color, thickness=2)
                
                if len(right_fitx_viz) > 0 and len(ploty_viz) > 0:
                    right_points = np.array([np.transpose(np.vstack([right_fitx_viz, ploty_viz]))], dtype=np.int32)
                    cv2.polylines(warped_lane_img, right_points, isClosed=False, color=color, thickness=2)

            warped_lanes_display_resized = cv2.resize(warped_lane_img, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow('Warped Lanes Detected', warped_lanes_display_resized)
        
        # classify lanes and get current lane
        warped_width = binary_warped.shape[1]

        # SET TO FALSE TEMPORARILY
        lane_info = get_current_lane(lanes, vehicle_center=None, image_width=img.shape[1], warped_width=warped_width, debug=False)
        current_lane_data = lane_info['current_lane']
        all_lanes = lane_info['all_lanes']

        # extract left and right fitx
        # Use current lane or fallback to first
        if current_lane_data:
            lane_data = current_lane_data['lane_data']
            if debug_display:
                print(f"[MAIN] Using selected lane: id={current_lane_data['lane_id']}, class={current_lane_data['lane_class']}, pos_in_lane={current_lane_data['position_in_lane']:.2f}")
        else:
            first_lane = list(all_lanes.values())[0] if all_lanes else None
            if first_lane:
                lane_data = first_lane['lane_data']
                print("[MAIN] No lane selected, using first detected lane")
            else:
                print("[MAIN] No lane data available")
                result = img.copy()
                metrics = {
                    'valid': False,
                    'current_lane': None,
                    'all_lanes': None,
                    'error': 'No lane data'
                }
                return result, metrics
        
        # extract data
        ploty = lane_data['ploty']
        left_fitx = lane_data['left_fitx']
        right_fitx = lane_data['right_fitx']
        left_fit = lane_data['left_fit']
        right_fit = lane_data['right_fit']
        
        # drawl all detected lanes
        result = draw_multiple_lanes_overlay(img, binary_warped, Minv, lanes, all_lanes_classified=all_lanes)
        
        if debug_display:
            warped_lane_img_debug = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
            h, w = binary_warped.shape[:2]
            
            for lane_idx, lane in enumerate(lanes):
                left_fitx_viz = lane['left_fitx'].astype(np.int32)
                right_fitx_viz = lane['right_fitx'].astype(np.int32)
                ploty_viz = lane['ploty'].astype(np.int32)
                lane_id = lane['lane_id']
                
                left_fitx_viz = np.clip(left_fitx_viz, 0, w - 1)
                right_fitx_viz = np.clip(right_fitx_viz, 0, w - 1)
                ploty_viz = np.clip(ploty_viz, 0, h - 1)
                
                is_current = current_lane_data and lane_id == current_lane_data['lane_id']
                color = (0, 255, 0) if is_current else (255, 255, 255)
                thickness = 3 if is_current else 2
                
                if len(left_fitx_viz) > 0 and len(ploty_viz) > 0:
                    left_points = np.array([np.transpose(np.vstack([left_fitx_viz, ploty_viz]))], dtype=np.int32)
                    cv2.polylines(warped_lane_img_debug, left_points, isClosed=False, color=color, thickness=thickness)
                
                if len(right_fitx_viz) > 0 and len(ploty_viz) > 0:
                    right_points = np.array([np.transpose(np.vstack([right_fitx_viz, ploty_viz]))], dtype=np.int32)
                    cv2.polylines(warped_lane_img_debug, right_points, isClosed=False, color=color, thickness=thickness)
                
                if len(left_fitx_viz) > 0 and len(right_fitx_viz) > 0:
                    label_x = int((left_fitx_viz[-1] + right_fitx_viz[-1]) / 2)
                    label_y = int(ploty_viz[-1])
                    label = f"L{lane_id}"
                    if is_current:
                        label += " *TRACKED*"
                    cv2.putText(warped_lane_img_debug, label, (label_x - 30, label_y + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            cv2.line(warped_lane_img_debug, (w//2, 0), (w//2, h), (0, 0, 255), 2)
            cv2.putText(warped_lane_img_debug, "VEHICLE CENTER", (w//2 + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            warped_lanes_display_resized = cv2.resize(warped_lane_img_debug, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow('Warped Lanes Detected', warped_lanes_display_resized)
        
        # Calculate metrics
        lane_metrics = calculate_curvature_and_deviation(left_fit, right_fit, left_fitx, right_fitx, ploty, binary_warped.shape, speed=speed, lookahead_m=lookahead_m)

        if lane_metrics is None:
            print("[METRICS] Lane detection metrics calculation returned None values")
            lane_metrics = {
                'deviation': None,
                'deviation_ahead': None,
                'lookahead_m': None,
                'curvature_radius': None,
                'lane_center': None,
                'vehicle_center': None,
                'lane_width': None,
            }
        valid = lane_metrics['deviation_ahead'] is not None
        deviation = lane_metrics['deviation']
        deviation_ahead = lane_metrics['deviation_ahead']
        curvature_radius = lane_metrics['curvature_radius']

        if debug_display and valid:
            print(f"[METRICS] dev={deviation:.4f}m, dev_ahead={deviation_ahead:.4f}m @ {lane_metrics['lookahead_m']:.1f}m")
            print(f"[METRICS] lane_center={lane_metrics['lane_center']:.1f}px, vehicle_center={lane_metrics['vehicle_center']:.1f}px, lane_width={lane_metrics['lane_width']:.1f}px")
            print(f"[METRICS] curvature radius={curvature_radius:.1f}m")

        result = add_text_overlay(result, curvature_radius, deviation, deviation_ahead, avg_brightness, speed)

        if debug_display:
            debug_y = 150
            if valid:
                cv2.putText(result, f"Dev: {deviation:.3f}m", (10, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(result, f"Dev ahead ({lane_metrics['lookahead_m']:.1f}m): {deviation_ahead:.3f}m", (10, debug_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if current_lane_data:
                cv2.putText(result, f"Lane: {current_lane_data['lane_class']} (ID:{current_lane_data['lane_id']}) pos={current_lane_data['position_in_lane']:.2f}", 
                           (10, debug_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result, f"Curv R:{curvature_radius:.0f}m" if valid else "Curv: N/A", 
                       (10, debug_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        metrics = {
            'valid': valid,
            **lane_metrics,
            'current_lane': current_lane_data,
            'all_lanes': all_lanes,
            'detected_num_lanes': detected_num_lanes
        }
        
        return result, metrics
        
    except Exception as e:
        print(f"Lane detection error CV: {e}")
        import traceback
        traceback.print_exc()
        result = img.copy()
        metrics = {
            'valid': False,
            'current_lane': None,
            'all_lanes': None,
            'error': str(e)
        }
        return result, metrics