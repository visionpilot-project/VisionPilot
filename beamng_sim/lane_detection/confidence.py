"""
Shared Features:
    1. Number of detected lane lines
    2. Length/Continuity of each detected lane line
    3. Proportion of ROI covered by detected lanes
    4. Lane Geometry, are the two lanes parallel, are they posittioned with a reasonable distance
    5. Constancy over time, how stable is the detection over multiple frames
"""
weights = {
    'num_lines': 0.2,
    'length': 0.2,
    'geometry': 0.4,
    # 'unet_specific': 0.2
}


def num_lane_lines(left_fitx, right_fitx):
    """
    Checks if the detection has both a left and right lane
    Args:
        left_fitx (np.array): x coordinates of the left lane line
        right_fitx (np.array): x coordinates of the right lane line
        
    Returns:
        int: number of detected lane lines (0, 1, or 2)
    """
    left_hand_detected = left_fitx is not None and len(left_fitx) > 0
    right_hand_detected = right_fitx is not None and len(right_fitx) > 0

    num_lanes = (1 if left_hand_detected else 0) + (1 if right_hand_detected else 0)

    if num_lanes == 2:
        num_lanes_score = 1.0
    elif num_lanes == 1:
        num_lanes_score = 0.5
    else:
        num_lanes_score = 0.0

    return num_lanes_score

def lane_length_continuity(ploty, left_fitx, right_fitx):
    """
    Computes the length and continuity of each detected lane line
    Args:
        ploty (np.array): y coordinates of the lane lines
        left_fitx (np.array): x coordinates of the left lane line
        right_fitx (np.array): x coordinates of the right lane line
        
    Returns:
        float: confidence score based on lane length and continuity
    """
    if ploty is None or len(ploty) == 0:
        return 0.0
    


#def lane_coverage():
    # Placeholder for lane coverage function

def lane_geometry(left_fitx, right_fitx, ploty):
    """
    Evaluates the geometry of the detected lanes
    Args:
        left_fitx (np.array): x coordinates of the left lane line
        right_fitx (np.array): x coordinates of the right lane line
        
    Returns:
        float: confidence score based on lane geometry
    """

def temporal_consistency(current_fit, previous_fit):
    """
    Assesses the stability of lane detection over multiple frames
    Args:
        current_fit (tuple): current frame's lane line polynomial coefficients
        previous_fit (tuple): previous frame's lane line polynomial coefficients
    Returns:
        float: confidence score based on temporal consistency
    """

def compute_shared_confidence(left_fitx, right_fitx, ploty, current_fit=None, previous_fit=None):

    num_lines_score = num_lane_lines(left_fitx, right_fitx)
    length_continuity_score = lane_length_continuity(ploty, left_fitx, right_fitx)
    geometry_score = lane_geometry(left_fitx, right_fitx, ploty)
    temporal_score = temporal_consistency(current_fit, previous_fit)


    return {
        'num_lines_score': num_lines_score,
        'length_continuity_score': length_continuity_score,
        'geometry_score': geometry_score,
        'temporal_score': temporal_score,
    }

def compute_confidence_unet(left_fitx, right_fitx, ploty):
    shared_score = compute_shared_confidence(left_fitx, right_fitx, ploty)

    # Add UNet-specific metrics

    confidence = (
        weights['num_lines'] * shared_score['num_lines_score'] +
        weights['length'] * shared_score['length_continuity_score'] +
        weights['geometry'] * shared_score['geometry_score']
        # weights['unet_specific'] * unet_specific_score
    ) / sum(weights.values())

    return min(max(confidence, 0.0), 1.0)


def compute_confidence_cv(left_fitx, right_fitx, ploty):
    shared_score = compute_shared_confidence(left_fitx, right_fitx, ploty)

    # Add CV-specific metrics
    
    # Combine scores
    confidence = (
        weights['num_lines'] * shared_score['num_lines_score'] +
        weights['length'] * shared_score['length_continuity_score'] +
        weights['geometry'] * shared_score['geometry_score']
        # weights['cv_specific'] * cv_specific_score
    ) / sum(weights.values())
    
    return min(max(confidence, 0.0), 1.0)