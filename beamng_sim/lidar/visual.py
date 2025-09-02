import os
import sys
import numpy as np
import pickle
import open3d as o3d

def load_lidar_data(file_path):
    with open(file_path, 'rb') as f:
        point_clouds = pickle.load(f)
    return point_clouds

def visualize_point_cloud_o3d(point_cloud):
    if not isinstance(point_cloud, np.ndarray):
        point_cloud = np.array(point_cloud)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
    
    o3d.visualization.draw_geometries([pcd, coordinate_frame], 
                                     window_name="LiDAR Point Cloud",
                                     width=1024, height=768,
                                     point_show_normal=False)

def visualize_multiple_point_clouds(file_path, cloud_indices=None, combine=False):
    point_clouds = load_lidar_data(file_path)
    print(f"Loaded {len(point_clouds)} point clouds from {file_path}")
    
    if cloud_indices is None:
        if len(point_clouds) >= 3:
            cloud_indices = [0, len(point_clouds)//2, len(point_clouds)-1]
        else:
            cloud_indices = range(len(point_clouds))
    
    if combine:
        combined_cloud = []
        for idx in cloud_indices:
            if idx < len(point_clouds):
                combined_cloud.extend(point_clouds[idx])
        
        print(f"Visualizing combined point cloud with {len(combined_cloud)} points")
        visualize_point_cloud_o3d(combined_cloud)
    else:
        for idx in cloud_indices:
            if idx < len(point_clouds):
                print(f"Visualizing point cloud {idx} with {len(point_clouds[idx])} points")
                visualize_point_cloud_o3d(point_clouds[idx])

def find_lidar_data_files():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_files = [f for f in os.listdir(current_dir) if f.endswith('.pkl') and 'lidar' in f.lower()]
    return pickle_files

if __name__ == "__main__":
    lidar_files = find_lidar_data_files()
    
    if not lidar_files:
        print("No LiDAR data files found. Please run data collection first.")
        sys.exit(1)
    
    print("Available LiDAR data files:")
    for i, file in enumerate(lidar_files):
        print(f"{i+1}. {file}")
    
    default_file = None
    for file in lidar_files:
        if file == "lidar_data.pkl":
            default_file = file
            break
        if "checkpoint" in file:
            checkpoint_files = [f for f in lidar_files if "checkpoint" in f]
            checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            default_file = checkpoint_files[-1]  # Get the latest checkpoint
    
    if not default_file and lidar_files:
        default_file = lidar_files[0]
    
    print(f"\nUsing file: {default_file}")
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), default_file)
    
    # Visualize options
    print("\nVisualization options:")
    print("1. View the latest point cloud")
    print("2. View first, middle and last point clouds")
    print("3. View combined point cloud (all points)")
    
    choice = input("Enter your choice (1-3) [1]: ") or "1"
    
    if choice == "1":
        # Show just the last point cloud
        point_clouds = load_lidar_data(file_path)
        visualize_point_cloud_o3d(point_clouds[-1])
    elif choice == "2":
        # Show first, middle and last point clouds
        visualize_multiple_point_clouds(file_path)
    elif choice == "3":
        # Show combined point cloud
        visualize_multiple_point_clouds(file_path, combine=True)
    else:
        print("Invalid choice, using option 1")
        point_clouds = load_lidar_data(file_path)
        visualize_point_cloud_o3d(point_clouds[-1])
