import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

"""
Utility function for collecting lidar point cloud data, to later be used for processing
"""

def collect_lidar_data(beamng, lidar_data):
    
    if lidar_data is None:
        print("LiDAR data is None")
        return []
    
    print(f"LiDAR data type: {type(lidar_data)}")
    print(f"LiDAR data keys: {lidar_data.keys() if isinstance(lidar_data, dict) else 'Not a dict'}")
    
    readings_data = lidar_data
    
    if isinstance(readings_data, dict):
        for key in readings_data.keys():
            val = readings_data[key]
            if isinstance(val, list):
                print(f"  - {key}: list with {len(val)} items")
            elif isinstance(val, np.ndarray):
                print(f"  - {key}: numpy array with shape {val.shape}")
            else:
                print(f"  - {key}: {type(val)}")
    
    point_cloud = readings_data.get("pointCloud", [])
    
    if isinstance(point_cloud, np.ndarray):
        print(f"Point cloud is numpy array with shape: {point_cloud.shape}")
    else:
        print(f"Point cloud type: {type(point_cloud)}, length: {len(point_cloud) if hasattr(point_cloud, '__len__') else 'N/A'}")

    return point_cloud