from beamng_sim.lidar.lidar import collect_lidar_data
import numpy as np

from .lane_boundry import detect_lane_boundaries
from .visualizer import LidarVisualizer3D, SimpleLidarViewer, HAS_OPEN3D

bin_size = 1.0
y_min, y_max = 60, 180  # LiDAR Y range is 60-170m

_visualizer = None

def get_visualizer():
    """Get or create visualizer (lazy init)."""
    global _visualizer
    if _visualizer is None:
        if HAS_OPEN3D:
            _visualizer = LidarVisualizer3D()
        else:
            _visualizer = SimpleLidarViewer()
    return _visualizer

def process_frame(lidar_sensor, beamng, speed, debug_window=None, vehicle=None):
    
    try:
        lidar_data = lidar_sensor.poll()
        if lidar_data is None:
            print("Warning: LiDAR sensor returned None")
            return []
            
        point_cloud = collect_lidar_data(beamng, lidar_data)

        try:
            point_cloud_len = len(point_cloud) if hasattr(point_cloud, '__len__') else 0
        except (TypeError, ValueError):
            point_cloud_len = 0
            
        if point_cloud_len == 0:
            print("Warning: Empty LiDAR point cloud")
            return []

        filtered_points = point_cloud
        
        import random
        if random.random() < 0.1:
            pc_array = np.array(filtered_points)
            print(f"LiDAR: {len(filtered_points)} points | Y: [{pc_array[:, 1].min():.1f}, {pc_array[:, 1].max():.1f}]")
        
        # Get car position for centering visualization
        car_position = None
        if vehicle is not None:
            try:
                vehicle.poll_sensors()
                if 'pos' in vehicle.state:
                    car_position = vehicle.state['pos']
            except:
                pass
        
        visualizer = get_visualizer()
        visualizer.update(filtered_points, car_position=car_position)
        
        # TODO: Optimize with numpy vectorization
        return []
        
    except Exception as e:
        print(f"LiDAR processing error: {e}")
        import traceback
        traceback.print_exc()
        return []