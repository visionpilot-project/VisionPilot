from beamng_sim.lane_detection.metrics import process_deviation

def fuse_lane_metrics(cv_metrics, cv_conf, dl_metrics, dl_conf, method_name="DL"):
    """
    Fuse lane detection metrics from CV and Deep Learning methods based on their confidence scores.
    CV is weighted more heavily (80%) as it is generally more trustworthy, while SCNN/DL (20%) is still used.
    Smoothing is applied AFTER fusion to the raw deviation values.
    
    Args:
        cv_metrics (dict): Metrics from the CV-based lane detection.
        cv_conf (float): Confidence score for the CV-based detection.
        dl_metrics (dict): Metrics from the DL-based lane detection (UNet or SCNN).
        dl_conf (float): Confidence score for the DL-based detection.
        method_name (str): Name of DL method for logging ("UNet" or "SCNN")
    Returns:
        dict: Fused lane detection metrics.
    """
    cv_weight = 0.80
    dl_weight = 0.20
    
    if cv_conf == 0.0 and dl_conf == 0.0:
        return{
            'left_curverad': 0.0,
            'right_curverad': 0.0,
            'deviation': 0.0,
            'smoothed_deviation': 0.0,
            'effective_deviation': 0.0,
            'lane_center': 0.0,
            'vehicle_center': 0.0,
            'confidence': 0.0,
        }
    
    def weighted(key):
        cv_val = cv_metrics.get(key, 0.0)
        dl_val = dl_metrics.get(key, 0.0)
        
        cv_val = 0.0 if cv_val is None else float(cv_val)
        dl_val = 0.0 if dl_val is None else float(dl_val)
        
        fused_val = (cv_val * cv_weight) + (dl_val * dl_weight)
        return fused_val

    # Fuse the RAW deviation values first (before any smoothing)
    fused_raw_deviation = weighted('deviation')
    
    # Now apply smoothing and processing to the fused raw deviation
    smoothed_dev, effective_dev = process_deviation(fused_raw_deviation, alpha=0.5, dead_zone=0.12, max_dev=2.0)

    fused = {
        'left_curverad': weighted('left_curverad'),
        'right_curverad': weighted('right_curverad'),
        'deviation': fused_raw_deviation,
        'smoothed_deviation': smoothed_dev,
        'effective_deviation': effective_dev,
        'lane_center': weighted('lane_center'),
        'vehicle_center': weighted('vehicle_center'),
        'confidence': (cv_conf * cv_weight + dl_conf * dl_weight)
    }
    print(f"Fused metrics (CV 80%, {method_name} 20%): deviation={fused_raw_deviation:.3f}, smoothed={smoothed_dev:.3f}, effective={effective_dev:.3f}")
    return fused