import pickle
import numpy as np
import open3d as o3d

file_path = "beamng_sim/lidar/lidar_data.pkl"
with open(file_path, 'rb') as f:
    point_clouds = pickle.load(f)
    color_clouds = [np.ones_like(cloud) for cloud in point_clouds]  # white color for all points

all_points = []
for cloud in point_clouds:
    all_points.extend(cloud)

pcd = o3d.geometry.PointCloud()

all_colors = []
for cloud, color_cloud in zip(point_clouds, color_clouds):
    all_points.extend(cloud)
    all_colors.extend(color_cloud)

pcd.points = o3d.utility.Vector3dVector(np.array(all_points))
pcd.colors = o3d.utility.Vector3dVector(np.array(all_colors))
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
o3d.visualization.draw_geometries([pcd, coordinate_frame], window_name="LiDAR Point Cloud (Merged)", width=1024, height=768, point_show_normal=False)