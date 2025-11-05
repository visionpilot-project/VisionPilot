import json
import numpy as np
from pathlib import Path
import open3d as o3d



class LidarPreprocessor:
    """Preprocesses raw LiDAR data."""
    
    def __init__(self, raw_data_dir='../data/raw', output_dir='../data/processed'):
        """
        Initialize LiDAR preprocessor.
        
        Args:
            raw_data_dir: Directory containing raw LiDAR data
            output_dir: Directory to save processed data
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def SOR_filter(self, points, mean_k=10, thresh=0.001):
        """
        Statistical Outlier Removal to filter out noise. 
        It removes points that dont fit the local structure or neighboring points.
        """
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        outlier_filter = pcd.remove_statistical_outlier(mean_k, thresh)
        filtered_points = np.asarray(outlier_filter[0].points)
        return filtered_points
    
    def ROR_filter(self, points, radius=1.0, min_neighbors=5):
        """
        Radius Outlier Removal to filter out points when
        they dont have a min amount of neighbors in the specified radius.

        Args:
            points: Input point cloud
            radius: Search radius
            min_neighbors: Minimum neighbors within radius

        Returns:
            Filtered point cloud
        """
        try:
            import open3d as o3d
        except ImportError:
            print("⚠️  open3d required for ROR filter")
            return points
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        outlier_filter = pcd.remove_radius_outlier(min_neighbors, radius)
        filtered_points = np.asarray(outlier_filter[0].points)
        return filtered_points
    
    def passthrough_filter(self, points, x_limits=None, y_limits=None, z_limits=None):
        """
        Passthrough filter to limit points within specified axis limits.

        Args:
            points: Input point cloud
            x_limits: Tuple (min, max) for x-axis
            y_limits: Tuple (min, max) for y-axis
            z_limits: Tuple (min, max) for z-axis

        Returns:
            Filtered point cloud
        """
        filtered_points = points.copy()
        
        if x_limits is not None:
            mask = (filtered_points[:, 0] >= x_limits[0]) & (filtered_points[:, 0] <= x_limits[1])
            filtered_points = filtered_points[mask]

        if y_limits is not None:
            mask = (filtered_points[:, 1] >= y_limits[0]) & (filtered_points[:, 1] <= y_limits[1])
            filtered_points = filtered_points[mask]

        if z_limits is not None:
            mask = (filtered_points[:, 2] >= z_limits[0]) & (filtered_points[:, 2] <= z_limits[1])
            filtered_points = filtered_points[mask]

        return filtered_points
    
    def process_frame(self, points, use_sor=False, sor_params=None, use_ror=False, ror_params=None, use_passthrough=False, passthrough_params=None):
        """
        Process a single frame of LiDAR data, applying selected filters.
        
        Args:
            points: Input point cloud (numpy array, shape: (N, 3))
            use_sor: Apply Statistical Outlier Removal
            sor_params: Dict with SOR params (mean_k, thresh)
            use_ror: Apply Radius Outlier Removal
            ror_params: Dict with ROR params (radius, min_neighbors)
            use_passthrough: Apply passthrough filter
            passthrough_params: Dict with axis limits (x_limits, y_limits, z_limits)
        
        Returns:
            Filtered point cloud (numpy array)
        """
        filtered_points = points.copy()
        
        if use_sor and sor_params:
            filtered_points = self.SOR_filter(filtered_points, **sor_params)
        
        if use_ror and ror_params:
            filtered_points = self.ROR_filter(filtered_points, **ror_params)
        
        if use_passthrough and passthrough_params:
            filtered_points = self.passthrough_filter(filtered_points, **passthrough_params)
        
        return filtered_points
