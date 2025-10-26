"""
Simple, fast 3D LiDAR point cloud visualizer using open3d
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("⚠️  open3d not installed. Install with: pip install open3d")


class LidarVisualizer3D:
    """Fast 3D visualization of LiDAR point clouds."""
    
    def __init__(self):
        if not HAS_OPEN3D:
            self.vis = None
            return
            
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="LiDAR Point Cloud", width=600, height=500)
        
        ctr = self.vis.get_view_control()
        ctr.set_zoom(0.6)
        ctr.set_front([0, -1, -0.3])  # Looking forward
        ctr.set_lookat([0, 0, 0])     # Focus on car origin
        ctr.set_up([0, 0, 1])         # Z is up
        
        self.pcd = None
        self.frame_count = 0
        print("LiDAR 3D visualizer initialized (open3d) - car-centered view")
    
    def update(self, points, car_position=None):
        """Update visualization with new points.
        
        Args:
            points: LiDAR point cloud array
            car_position: (x, y, z) position of the car in world coordinates
        """
        if not HAS_OPEN3D or self.vis is None:
            return
        
        try:
            pcd = o3d.geometry.PointCloud()
            points_array = np.asarray(points, dtype=np.float32)
            
            if car_position is not None:
                car_pos = np.array(car_position, dtype=np.float32)
                points_array = points_array - car_pos
            
            pcd.points = o3d.utility.Vector3dVector(points_array)
            
            if len(points) > 0:
                distances = np.linalg.norm(points_array, axis=1)
                d_min, d_max = distances.min(), distances.max()
                d_normalized = (distances - d_min) / (d_max - d_min + 1e-6)
                colors = cm.plasma(d_normalized)[:, :3]  # Red (close) to yellow (far)
                pcd.colors = o3d.utility.Vector3dVector(colors)
            
            if self.pcd is None:
                self.vis.add_geometry(pcd)
            else:
                self.vis.clear_geometries()
                self.vis.add_geometry(pcd)
            
            self.pcd = pcd
            
            self.vis.poll_events()
            self.vis.update_renderer()
            self.frame_count += 1
        
        except Exception as e:
            print(f"Visualization error: {e}")
    
    def close(self):
        """Close the visualizer."""
        if self.vis is not None:
            self.vis.destroy_window()


class SimpleLidarViewer:
    """Ultra-simple fallback using matplotlib 3D scatter."""
    
    def __init__(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        self.fig = plt.figure(figsize=(7, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.suptitle("LiDAR Point Cloud", fontsize=12)
        
        plt.ion()
        print("LiDAR simple 3D viewer initialized (matplotlib)")
    
    def update(self, points, car_position=None):
        """Update visualization.
        
        Args:
            points: LiDAR point cloud array
            car_position: (x, y, z) position of the car in world coordinates
        """
        try:
            self.ax.clear()
            
            if len(points) > 0:
                pts = np.asarray(points, dtype=np.float32)
                
                if car_position is not None:
                    car_pos = np.array(car_position, dtype=np.float32)
                    pts = pts - car_pos
                
                distances = np.linalg.norm(pts, axis=1)
                
                self.ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], 
                               c=distances, cmap='plasma', s=1, alpha=0.6)
                self.ax.set_xlabel('X (m)')
                self.ax.set_ylabel('Y (m)')
                self.ax.set_zlabel('Z (m)')
            
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.001)
        
        except Exception as e:
            print(f"Viewer error: {e}")
    
    def close(self):
        import matplotlib.pyplot as plt
        plt.close(self.fig)
