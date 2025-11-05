from beamng_sim.lidar.lidar import collect_lidar_data
import numpy as np

from .lane_boundry import detect_lane_boundaries
from .visualizer import LidarVisualizer3D
from .lidar_preprocessing import LidarPreprocessor

_visualizer = None
_preprocessor = LidarPreprocessor()

def get_visualizer():
    """Get or create visualizer (lazy init)."""
    global _visualizer
    if _visualizer is None:
        try:
            _visualizer = LidarVisualizer3D()
            print("Visualizer initialized (open3d)")
        except Exception as e:
            print(f"Failed to initialize open3d visualizer: {e}")
    return _visualizer

def process_frame(lidar_sensor, beamng, speed, debug_window=None, vehicle=None, car_position=None, car_direction=None):
    
    try:
        lidar_data = lidar_sensor.poll()
        if lidar_data is None:
            print("Warning: LiDAR sensor returned None")
            return []
            
        point_cloud = collect_lidar_data(beamng, lidar_data)

        if len(point_cloud) == 0:
            print("Warning: Empty LiDAR point cloud")
            return []

        car_yaw = None
        if car_direction is not None:
            car_yaw = np.arctan2(car_direction[1], car_direction[0])
            print(f"Car direction: {car_direction}, Car yaw: {np.degrees(car_yaw):.1f} degrees")

        if car_position is not None:
            car_x, car_y, car_z = car_position
            x_limits = (car_x - 10.0, car_x + 10.0)
            y_limits = (car_y, car_y + 60.0)
            z_limits = (car_z - 4.0, car_z + 4.0)
        else:
            x_limits = (None, None)
            y_limits = (None, None)
            z_limits = (None, None)

        filtered_points = _preprocessor.process_frame(
            point_cloud,
            use_passthrough=True,
            passthrough_params={
                'x_limits': x_limits,      # Dynamic left/right of car
                'y_limits': y_limits,      # Dynamic ahead of car
                'z_limits': z_limits       # Dynamic Z window around car
            }
        )

        print(f"After passthrough: {len(filtered_points)} points (from {len(point_cloud)})")
        
        # filtered_points = _preprocessor.process_frame(
        #     filtered_points,
        #     use_ror=True,
        #     ror_params={'radius': 0.5, 'min_neighbors': 3}
        # )
        
        if len(filtered_points) == 0:
            print("Warning: No points after filtering")
            print(f"  Point cloud range: X=[{point_cloud[:, 0].min():.1f}, {point_cloud[:, 0].max():.1f}]")
            print(f"  Point cloud range: Y=[{point_cloud[:, 1].min():.1f}, {point_cloud[:, 1].max():.1f}]")
            print(f"  Point cloud range: Z=[{point_cloud[:, 2].min():.1f}, {point_cloud[:, 2].max():.1f}]")
            return []
        
        visualizer = get_visualizer()
        visualizer.update(filtered_points, car_position=car_position, car_yaw=car_yaw)
        
        # todo Calculate and visualize lane boundaries
        boundaries = detect_lane_boundaries(filtered_points)
        
        return boundaries
        
    except Exception as e:
        print(f"LiDAR processing error: {e}")
        return []