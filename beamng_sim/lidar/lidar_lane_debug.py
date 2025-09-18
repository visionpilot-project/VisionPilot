import open3d as o3d
import numpy as np

class LiveLidarDebugWindow:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="LiDAR Road Boundary Debug", width=1024, height=768)
        self.pcd = o3d.geometry.PointCloud()
        self.line_set = o3d.geometry.LineSet()
        self.vis.add_geometry(self.pcd)
        self.vis.add_geometry(self.line_set)
        opt = self.vis.get_render_option()
        opt.background_color = np.array([0, 0, 0])

    def update(self, points, boundaries):
        # Color points by y (height)
        points = np.array(points)
        if len(points) == 0:
            return
        y_vals = points[:, 1]
        y_min, y_max = y_vals.min(), y_vals.max()
        y_norm = (y_vals - y_min) / (y_max - y_min + 1e-6)
        colors = np.stack([y_norm, np.zeros_like(y_norm), 1 - y_norm], axis=1)  # Red to Blue

        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

        # Build boundary lines
        line_points = []
        lines = []
        line_colors = []
        idx = 0
        for b in boundaries:
            y_start, y_end = b['y_bin']
            left = b['left']
            right = b['right']
            mean_z = b['mean_road_height']
            # Left boundary (green)
            line_points.append([left, y_start, mean_z])
            line_points.append([left, y_end, mean_z])
            lines.append([idx, idx+1])
            line_colors.append([0,1,0])
            idx += 2
            # Right boundary (red)
            line_points.append([right, y_start, mean_z])
            line_points.append([right, y_end, mean_z])
            lines.append([idx, idx+1])
            line_colors.append([1,0,0])
            idx += 2

        if line_points:
            self.line_set.points = o3d.utility.Vector3dVector(np.array(line_points))
            self.line_set.lines = o3d.utility.Vector2iVector(lines)
            self.line_set.colors = o3d.utility.Vector3dVector(line_colors)
        else:
            self.line_set.points = o3d.utility.Vector3dVector()
            self.line_set.lines = o3d.utility.Vector2iVector()
            self.line_set.colors = o3d.utility.Vector3dVector()

        self.vis.update_geometry(self.pcd)
        self.vis.update_geometry(self.line_set)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()