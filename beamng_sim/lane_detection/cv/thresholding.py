import numpy as np
import cv2


def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dx = 1 if orient == 'x' else 0
    dy = 0 if orient == 'x' else 1
    sobel = cv2.Sobel(gray, cv2.CV_64F, dx, dy)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel)) 
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary


def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel)) 
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return mag_binary


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    dir_binary = np.zeros_like(grad_dir)
    dir_binary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    return dir_binary


def gradient_thresholds(image, ksize=3, avg_brightness=None):
    x_low, x_high = 40, 120
    y_low, y_high = 40, 120
    mag_low, mag_high = 50, 120
    
    if avg_brightness is not None:
        if avg_brightness < 80:
            x_low = 30
            y_low = 30
            mag_low = 40
        elif avg_brightness > 200:
            x_high = 160
            y_high = 160
            mag_high = 160
    
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(x_low, x_high))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(y_low, y_high))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(mag_low, mag_high))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) | (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    return combined


def color_threshold(image, avg_brightness=None):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    w_h_min, w_h_max = 0, 180
    w_s_min, w_s_max = 0, 50
    w_v_min, w_v_max = 160, 255

    y_h_min, y_h_max = 10, 45
    y_s_min, y_s_max = 60, 255
    y_v_min, y_v_max = 100, 255

    s_h_min, s_h_max = 0, 180
    s_s_min, s_s_max = 0, 20
    s_v_min, s_v_max = 110, 150


    if not hasattr(color_threshold, "brightness_history"):
        color_threshold.brightness_history = []

    if avg_brightness is not None:
        color_threshold.brightness_history.append(avg_brightness)
        if len(color_threshold.brightness_history) > 5:
            color_threshold.brightness_history.pop(0)
            
        avg_recent = np.mean(color_threshold.brightness_history)
        variance = np.var(color_threshold.brightness_history) if len(color_threshold.brightness_history) > 1 else 0
        
        print(f"Avg brightness: {avg_brightness:.1f}, Recent avg: {avg_recent:.1f}, Variance: {variance:.1f}")
        
        #is_stable_lighting = variance < 50

        # Implement adaptive thresholding based on lighting stability
        
        if avg_recent > 200:  # Very bright conditions (direct sunlight)
            w_s_max = 25
            w_v_min = 200
            y_s_min = 100
            
        elif avg_recent > 170:
            w_v_min = 200
            w_s_max = 20
            
        elif 100 < avg_recent < 170:
            w_v_min = 200
            w_s_max = 40

        elif 70 < avg_recent <= 100:
            w_v_min = 150
            w_s_max = 42
            s_v_max = 160

        elif avg_brightness <= 70:  # Low light conditions
            w_v_min = 120
            w_s_max = 45
            y_v_min = 90
            y_s_min = 50
            s_v_max = 150

    # Apply white mask
    white_lower = np.array([w_h_min, w_s_min, w_v_min])
    white_upper = np.array([w_h_max, w_s_max, w_v_max])
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    
    # Apply yellow mask
    yellow_lower = np.array([y_h_min, y_s_min, y_v_min])
    yellow_upper = np.array([y_h_max, y_s_max, y_v_max])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    
    shadow_mask = np.zeros_like(white_mask)
    if avg_brightness is not None and 50 < avg_brightness < 150:
        shadow_lower = np.array([s_h_min, s_s_min, s_v_min])
        shadow_upper = np.array([s_h_max, s_s_max, s_v_max])
        shadow_mask = cv2.inRange(hsv, shadow_lower, shadow_upper)
    
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    
    if avg_brightness is not None and 60 < avg_brightness < 120:
        combined_mask = cv2.bitwise_or(combined_mask, shadow_mask)
    
    combined_pixels = np.sum(combined_mask)
    print(f"Combined color mask pixels: {combined_pixels}")
    
    binary = np.zeros_like(hsv[:,:,0])
    binary[combined_mask > 0] = 1
    
    return binary


def lab_hls_threshold(image, avg_brightness=None):
    """
    Multi-colorspace thresholding using LAB and HLS color spaces.
    Following CarND's empirical finding that combining multiple color spaces
    provides robust lane detection.
    
    Args:
        image (numpy array): RGB image
        avg_brightness (float): Average brightness for adaptive thresholding
    
    Returns:
        numpy array: Binary image with detected lane pixels
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel = lab[:,:,0]
    b_channel = lab[:,:,2]
    
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    
    # LAB L-channel: excellent for white lines in all lighting conditions
    if avg_brightness is not None:
        if avg_brightness > 200:  # Very bright
            l_thresh = (220, 255)
        elif avg_brightness > 170:  # Bright
            l_thresh = (210, 255)
        elif avg_brightness < 80:  # Dark
            l_thresh = (180, 255)
        else:  # Medium
            l_thresh = (200, 255)
    else:
        l_thresh = (200, 255)
    
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
    
    # LAB B-channel: good for yellow detection (yellow has high B values)
    if avg_brightness is not None:
        if avg_brightness > 200:  # Very bright
            b_thresh = (155, 200)
        elif avg_brightness < 80:  # Dark
            b_thresh = (145, 200)
        else:  # Medium to bright
            b_thresh = (150, 200)
    else:
        b_thresh = (150, 200)
    
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh[0]) & (b_channel <= b_thresh[1])] = 1
    
    # HLS S-channel: proven robust for lane detection (CarND's finding)
    # Saturation is good for colored lanes (yellow) in various lighting
    if avg_brightness is not None:
        if avg_brightness > 200:  # Very bright
            s_thresh = (100, 255)
        elif avg_brightness < 80:  # Dark
            s_thresh = (80, 255)
        else:  # Medium
            s_thresh = (90, 255)
    else:
        s_thresh = (90, 255)
    
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Combine all channels
    # L-channel for white, B-channel for yellow, S-channel for both
    combined = np.zeros_like(l_channel)
    combined[(l_binary == 1) | (b_binary == 1) | (s_binary == 1)] = 1
    
    return combined


def majority_vote(binaries, n_vote):
    """
    Majority voting system for combining multiple binary threshold results.
    Following CarND's approach: requires n_vote out of total filters to agree.
    
    Args:
        binaries (list): List of binary threshold results (numpy arrays)
        n_vote (int): Number of filters that must agree (threshold)
    
    Returns:
        numpy array: Binary image where pixels passed majority vote
    """
    stacked = np.stack(binaries, axis=-1)
    sum_binary = np.sum(stacked, axis=-1)
    
    vote_binary = np.zeros_like(sum_binary)
    vote_binary[sum_binary >= n_vote] = 1
    
    return vote_binary.astype(np.uint8)


def adaptive_majority_vote(image, avg_brightness, include_gradient=True):
    """
    Adaptive majority voting that adjusts features and voting threshold
    based on lighting conditions.
    
    Combines ALL color spaces for maximum robustness:
    - HSV: Original adaptive white/yellow detection with sophisticated brightness logic
    - LAB L-channel: White line detection
    - LAB B-channel: Yellow line detection  
    - HLS S-channel: Saturation-based detection
    - Gradient: Edge detection (optional)
    
    Following recommendations from comparison report:
    - Dark (<100): More permissive (3 out of 5-6 features)
    - Medium (100-180): Balanced (3 out of 5-6 features) 
    - Bright (>180): Stricter (4 out of 5-6 features)
    
    Args:
        image (numpy array): RGB image
        avg_brightness (float): Average brightness of the image
        include_gradient (bool): Whether to include gradient features
    
    Returns:
        numpy array: Binary image from majority voting
    """
    # 1. HSV Color threshold (your original sophisticated adaptive logic)
    hsv_binary = color_threshold(image, avg_brightness=avg_brightness)
    
    # 2. Extract individual LAB/HLS channels for voting
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel = lab[:,:,0]
    b_channel = lab[:,:,2]
    
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    
    # 3. LAB L-channel threshold (white detection)
    if avg_brightness > 200:  # Very bright
        l_thresh = (220, 255)
    elif avg_brightness > 170:  # Bright
        l_thresh = (210, 255)
    elif avg_brightness < 80:  # Dark
        l_thresh = (180, 255)
    else:  # Medium
        l_thresh = (200, 255)
    
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
    
    # 4. LAB B-channel threshold (yellow detection)
    if avg_brightness > 200:  # Very bright
        b_thresh = (155, 200)
    elif avg_brightness < 80:  # Dark
        b_thresh = (145, 200)
    else:  # Medium to bright
        b_thresh = (150, 200)
    
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh[0]) & (b_channel <= b_thresh[1])] = 1
    
    # 5. HLS S-channel threshold (saturation - both colors)
    if avg_brightness > 200:  # Very bright
        s_thresh = (100, 255)
    elif avg_brightness < 80:  # Dark
        s_thresh = (80, 255)
    else:  # Medium
        s_thresh = (90, 255)
    
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # 6. Gradient features if requested
    if include_gradient:
        grad_binary = gradient_thresholds(image, avg_brightness=avg_brightness)
    
    # Adaptive voting based on brightness
    # Build feature list: HSV + L + B + S + (optional gradient)
    if avg_brightness < 100:  # Dark conditions
        # More permissive: 3 out of 5 or 6
        # In dark, HSV is crucial, combine with LAB/HLS and gradient
        if include_gradient:
            features = [hsv_binary, l_binary, b_binary, s_binary, grad_binary]
            n_vote = 3
            print(f"Dark mode: voting 3/5 (HSV + LAB L/B + HLS S + Gradient)")
        else:
            features = [hsv_binary, l_binary, b_binary, s_binary]
            n_vote = 2
            print(f"Dark mode: voting 2/4 (HSV + LAB L/B + HLS S)")
        
    elif avg_brightness < 180:  # Medium lighting
        # Balanced: 3 out of 5 or 6
        # Standard approach: all color spaces vote
        if include_gradient:
            features = [hsv_binary, l_binary, s_binary, b_binary, grad_binary]
            n_vote = 3
            print(f"Medium mode: voting 3/5 (HSV + LAB L/B + HLS S + Gradient)")
        else:
            features = [hsv_binary, l_binary, s_binary, b_binary]
            n_vote = 2
            print(f"Medium mode: voting 2/4 (HSV + LAB L/B + HLS S)")
        
    else:  # Bright conditions (>180)
        # Stricter: 4 out of 5 or 6
        # In bright conditions, require more agreement to reduce false positives
        if include_gradient:
            features = [hsv_binary, l_binary, b_binary, s_binary, grad_binary]
            n_vote = 4
            print(f"Bright mode: voting 4/5 (HSV + LAB L/B + HLS S + Gradient)")
        else:
            features = [hsv_binary, l_binary, b_binary, s_binary]
            n_vote = 3
            print(f"Bright mode: voting 3/4 (HSV + LAB L/B + HLS S)")
    
    # Perform majority voting
    result = majority_vote(features, n_vote)
    
    print(f"Majority vote pixels: {np.sum(result)}")
    print(f"  HSV: {np.sum(hsv_binary)}, L: {np.sum(l_binary)}, B: {np.sum(b_binary)}, S: {np.sum(s_binary)}" + 
          (f", Grad: {np.sum(grad_binary)}" if include_gradient else ""))
    
    return result


def combine_thresholds(color_binary, gradient_binary, avg_brightness=None):
    combined_binary = np.zeros_like(color_binary)
    
    color_coverage = np.sum(color_binary)
    gradient_coverage = np.sum(gradient_binary)
    total_pixels = color_binary.size
    
    gradient_weight = 1.0
    color_weight = 1.0
    
    if avg_brightness is not None:
        if avg_brightness < 100:
            gradient_weight = 1.5
            color_weight = 0.7
        elif avg_brightness > 200:
            gradient_weight = 0.9
            color_weight = 1.3
        else:
            gradient_weight = 1.0
            color_weight = 1.1
    
    color_confidence = color_coverage / total_pixels
    gradient_confidence = gradient_coverage / total_pixels
    
    combined_binary[color_binary == 1] = 1
    
    color_pixels = np.sum(color_binary)
    print(f"Color binary pixels going into combine: {color_pixels}")
    print(f"Combined binary pixels after adding color: {np.sum(combined_binary)}")
    
    if avg_brightness is not None:
        if avg_brightness < 120:  # In darker conditions, rely more on gradients
            combined_binary[gradient_binary == 1] = 1
        elif avg_brightness < 180:  # Medium lighting
            combined_binary[(gradient_binary == 1) & (color_binary == 1)] = 1
        else:  # Very bright conditions
            combined_binary[(gradient_binary == 1) & (color_binary == 1)] = 1
    else:
        combined_binary[gradient_binary == 1] = 1
    
    combined_binary = (combined_binary > 0).astype(np.uint8)
    
    final_pixels = np.sum(combined_binary)
    print(f"Final combined binary pixels: {final_pixels}")
    
    return combined_binary


def apply_thresholds_with_voting(image, src_points=None, debug_display=False, use_gradient=True):
    """
    Apply thresholds using majority voting system (CarND approach).
    This is the NEW recommended approach that combines adaptive thresholding
    with robust majority voting.
    
    Args:
        image (numpy array): RGB image
        src_points (numpy array): Source points for ROI masking (optional)
        debug_display (bool): Whether to show debug windows
        use_gradient (bool): Whether to include gradient features in voting
    
    Returns:
        tuple: (combined_binary, avg_brightness)
    """
    if src_points is not None:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        src_poly = np.array(src_points, dtype=np.int32)
        cv2.fillPoly(mask, [src_poly], 1)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        avg_brightness = np.mean(gray[mask == 1])
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        avg_brightness = np.mean(gray)
    
    print(f"Average brightness: {avg_brightness:.1f}")
    
    combined_binary = adaptive_majority_vote(image, avg_brightness, include_gradient=use_gradient)
    
    if debug_display:
        hsv_binary = color_threshold(image, avg_brightness=avg_brightness)
        hsv_binary_uint8 = hsv_binary.astype(np.uint8)
        
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:,:,0]
        b_channel = lab[:,:,2]
        
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        
        # L-channel binary
        if avg_brightness > 200:
            l_thresh = (220, 255)
        elif avg_brightness > 170:
            l_thresh = (210, 255)
        elif avg_brightness < 80:
            l_thresh = (180, 255)
        else:
            l_thresh = (200, 255)
        
        l_binary = np.zeros_like(l_channel, dtype=np.uint8)
        l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
        
        # S-channel binary
        if avg_brightness > 200:
            s_thresh = (100, 255)
        elif avg_brightness < 80:
            s_thresh = (80, 255)
        else:
            s_thresh = (90, 255)
        
        s_binary = np.zeros_like(s_channel, dtype=np.uint8)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        
        # B-channel binary
        if avg_brightness > 200:
            b_thresh = (155, 200)
        elif avg_brightness < 80:
            b_thresh = (145, 200)
        else:
            b_thresh = (150, 200)
        
        b_binary = np.zeros_like(b_channel, dtype=np.uint8)
        b_binary[(b_channel >= b_thresh[0]) & (b_channel <= b_thresh[1])] = 1
        
        # Gradient binary
        if use_gradient:
            grad_binary = gradient_thresholds(image, avg_brightness=avg_brightness)
            grad_binary_uint8 = grad_binary.astype(np.uint8)
        
        debug_img = np.zeros((combined_binary.shape[0], combined_binary.shape[1], 3), dtype=np.uint8)
        
        # Magenta: HSV detection (original adaptive logic)
        debug_img[hsv_binary_uint8 == 1] = [255, 0, 255]
        
        # Red: L-channel (white detection)
        debug_img[(hsv_binary_uint8 == 0) & (l_binary == 1)] = [0, 0, 255]
        
        # Green: S-channel (saturation)
        debug_img[(hsv_binary_uint8 == 0) & (l_binary == 0) & (s_binary == 1)] = [0, 255, 0]
        
        # Blue: B-channel (yellow detection)
        debug_img[(hsv_binary_uint8 == 0) & (l_binary == 0) & (s_binary == 0) & (b_binary == 1)] = [255, 0, 0]
        
        # Yellow: Multiple overlaps
        debug_img[(hsv_binary_uint8 == 1) & ((l_binary == 1) | (s_binary == 1))] = [0, 255, 255]
        
        # Display individual channels
        hsv_display = np.dstack((hsv_binary_uint8, hsv_binary_uint8, hsv_binary_uint8)) * 255
        cv2.imshow('HSV Adaptive (Original)', hsv_display)
        
        l_display = np.dstack((l_binary, l_binary, l_binary)) * 255
        cv2.imshow('LAB L-channel (White)', l_display)
        
        s_display = np.dstack((s_binary, s_binary, s_binary)) * 255
        cv2.imshow('HLS S-channel (Saturation)', s_display)
        
        b_display = np.dstack((b_binary, b_binary, b_binary)) * 255
        cv2.imshow('LAB B-channel (Yellow)', b_display)
        
        if use_gradient:
            grad_display = np.dstack((grad_binary_uint8, grad_binary_uint8, grad_binary_uint8)) * 255
            cv2.imshow('Gradient Threshold', grad_display)
        
        combined_display = np.dstack((combined_binary, combined_binary, combined_binary)) * 255
        cv2.imshow('Majority Vote Result', combined_display)
        
        cv2.imshow('Feature Contributions (All Color Spaces)', debug_img)
    
    return combined_binary, avg_brightness


def apply_thresholds(image, src_points=None, debug_display=False):
    """
    LEGACY FUNCTION - apply_thresholds_with_voting() instead
    
    This function is kept for backward compatibility but the new majority voting
    approach (apply_thresholds_with_voting) is recommended for better robustness.
    
    Apply gradient and color thresholds to detect lane lines (old conditional logic).
    """

    if src_points is not None:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        src_poly = np.array(src_points, dtype=np.int32)
        cv2.fillPoly(mask, [src_poly], 1)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        avg_brightness = np.mean(gray[mask == 1])
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        avg_brightness = np.mean(gray)


    grad_binary = gradient_thresholds(image, avg_brightness=avg_brightness)
    color_binary = color_threshold(image, avg_brightness=avg_brightness)
    combined_binary = combine_thresholds(color_binary, grad_binary, avg_brightness=avg_brightness)
    
    if debug_display:
        # Make sure all binary images are uint8 for OpenCV compatibility
        color_binary_uint8 = color_binary.astype(np.uint8)
        grad_binary_uint8 = grad_binary.astype(np.uint8)
        
        debug_display = np.zeros((combined_binary.shape[0], combined_binary.shape[1], 3), dtype=np.uint8)
        
        debug_display[color_binary_uint8 == 1] = [0, 0, 255]
        
        debug_display[(color_binary_uint8 == 0) & (grad_binary_uint8 == 1)] = [0, 255, 0]
        
        debug_display[(color_binary_uint8 == 1) & (grad_binary_uint8 == 1)] = [0, 255, 255]
        
        final_vis = np.zeros_like(debug_display)
        # Combined binary should already be uint8 from previous fix
        final_vis[combined_binary == 1] = [255, 255, 255]
        
        grad_display = np.dstack((grad_binary_uint8, grad_binary_uint8, grad_binary_uint8)) * 255
        cv2.imshow('Gradient Threshold', grad_display)
        
        color_display = np.dstack((color_binary_uint8, color_binary_uint8, color_binary_uint8)) * 255
        cv2.imshow('Color Threshold', color_display)
        
        combined_display_array = combined_binary.astype(np.uint8)
        combined_display = np.dstack((combined_display_array, combined_display_array, combined_display_array)) * 255
        cv2.imshow('Lane Detection Combined', combined_display)

        cv2.imshow('Detection Method Contributions', debug_display)
        
        cv2.imshow('Final Output Pixels', final_vis)

    return combined_binary, avg_brightness
