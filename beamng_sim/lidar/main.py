from beamng_sim.lidar.lidar import collect_lidar_data

def process_frame(lidar_sensor, camera_detections, beamng):
    lidar_data = lidar.poll()
    point_cloud = collect_lidar_data(beamng, lidar_data)

    # Process Point Cloud
    pass