from beamng_sim.lidar.lidar import collect_lidar_data
import numpy as np


from .lane_boundry import detect_lane_boundaries
bin_size = 1.0
y_min, y_max = 0, 30

def process_frame(lidar_sensor, camera_detections, beamng, speed):
    lidar_data = lidar_sensor.poll()
    point_cloud = collect_lidar_data(beamng, lidar_data)

    filtered_points = []
    for point in point_cloud:
        x, y, z = point[:3]
        distance = (x**2 + y**2 + z**2) ** 0.5
        if speed <= 70 and distance > 60:
            continue
        else:
            filtered_points.append((x, y, z))
            
    boundaries = detect_lane_boundaries(filtered_points)
    return boundaries