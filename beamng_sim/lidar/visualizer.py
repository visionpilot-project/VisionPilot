"""
Simple, fast 3D LiDAR point cloud visualizer using open3d
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import open3d as o3d


class LidarVisualizer3D:
    """Fast 3D visualization of LiDAR point clouds."""
    
    def __init__(self):
            
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="LiDAR Point Cloud", width=600, height=500)
        
        ctr = self.vis.get_view_control()
        ctr.set_zoom(0.6)
        ctr.set_front([0, -1, -0.5])  # Looking forward from behind
        ctr.set_lookat([0, 0, 0])     # Focus on car origin
        ctr.set_up([0, 0, 1])         # Z is up
        
        self.pcd = None
        self.frame_count = 0
        self.skip_frames = 0
        self.camera_offset = np.array([0, 4, 5], dtype=np.float32)
        print("LiDAR 3D visualizer initialized (open3d) - third-person view behind car")
    
    def update(self, points, car_position=None, car_yaw=None):
        """Update visualization with new points.
        
        Args:
            points: LiDAR point cloud array (in world coordinates)
            car_position: (x, y, z) position of the car in world coordinates
            car_yaw: Yaw angle in radians for the car's heading
        """
        
        if self.frame_count % 2 != 0:
            self.frame_count += 1
            return
        
        try:
            points_array = np.asarray(points, dtype=np.float32)
            
            if len(points_array) == 0:
                return
            
            if car_position is not None:
                car_pos = np.array(car_position, dtype=np.float32)
                third_person_offset = np.array([0, -8, -5], dtype=np.float32)
                
                if car_yaw is not None:
                    cos_yaw = np.cos(car_yaw)
                    sin_yaw = np.sin(car_yaw)
                    rotated_x = third_person_offset[0] * cos_yaw - third_person_offset[1] * sin_yaw
                    rotated_y = third_person_offset[0] * sin_yaw + third_person_offset[1] * cos_yaw
                    third_person_offset = np.array([rotated_x, rotated_y, third_person_offset[2]], dtype=np.float32)
                    print(f"Visualizer - Yaw: {np.degrees(car_yaw):.1f}°, Original offset: [0, -8, -5], Rotated offset: {third_person_offset}")
                
                points_array = points_array - car_pos + third_person_offset

            if self.pcd is None:
                self.pcd = o3d.geometry.PointCloud()
                self.pcd.points = o3d.utility.Vector3dVector(points_array)
                distances = np.linalg.norm(points_array, axis=1)
                d_min, d_max = distances.min(), distances.max()
                d_normalized = (distances - d_min) / (d_max - d_min + 1e-6)
                colors = cm.plasma(d_normalized)[:, :3]
                self.pcd.colors = o3d.utility.Vector3dVector(colors)
                self.vis.add_geometry(self.pcd)
            else:
                self.pcd.clear()
                self.pcd.points = o3d.utility.Vector3dVector(points_array)
                distances = np.linalg.norm(points_array, axis=1)
                d_min, d_max = distances.min(), distances.max()
                d_normalized = (distances - d_min) / (d_max - d_min + 1e-6)
                colors = cm.plasma(d_normalized)[:, :3]
                self.pcd.colors = o3d.utility.Vector3dVector(colors)
                self.vis.update_geometry(self.pcd)
            
            self.vis.poll_events()
            self.vis.update_renderer()
            self.frame_count += 1
        
        except Exception as e:
            print(f"Visualization error: {e}")
            self.frame_count += 1
    
    def close(self):
        """Close the visualizer."""
        if self.vis is not None:
            self.vis.destroy_window()
