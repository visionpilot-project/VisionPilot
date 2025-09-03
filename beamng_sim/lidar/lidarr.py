import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from time import sleep
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Lidar
import math


def yaw_to_quat(yaw_deg):
    yaw = math.radians(yaw_deg)
    w = math.cos(yaw / 2)
    z = math.sin(yaw / 2)
    return (0.0, 0.0, z, w)


def collect_lidar_data():
    beamng = BeamNGpy('localhost', 64256, home=r'C:\Users\user\Documents\beamng-tech\BeamNG.tech.v0.36.4.0')
    beamng.open()
    
    scenario = Scenario('west_coast_usa', 'lidar_test')
    vehicle = Vehicle('ego_vehicle', model='etk800', licence='LIDAR')
    
    rot_city = yaw_to_quat(-133.506 + 180)

    # Street Spawn
    scenario.add_vehicle(vehicle, pos=(-730.212, 94.630, 118.517), rot_quat=rot_city)
    
    scenario.make(beamng)
    
    # Set simulator to 60hz temporal resolution
    beamng.settings.set_deterministic(60)
    
    beamng.scenario.load(scenario)
    beamng.scenario.start()
    
    # Create LiDAR sensor after scenario has started
    lidar = Lidar(
        "lidar1",
        beamng,
        vehicle,
        requested_update_time=0.01,
        is_using_shared_memory=True,
        is_360_mode=True,  # 360-degree mode
        vertical_angle=30,  # Vertical field of view
        vertical_resolution=256,  # Number of lasers/channels
        density=10,
        frequency=30,
        max_distance=60,
        pos=(0, 0.2, 1.8),
    )
    
    # Storage for collected point clouds
    all_point_clouds = []
    
    # Set up AI driver or manual control
    # vehicle.ai.set_mode("traffic")  # Use AI to drive the vehicle
    
    import time
    print("Collecting LiDAR data for 1 minute...")
    start_time = time.time()
    frame_count = 0
    while time.time() - start_time < 60:
        beamng.control.step(10)
        readings_data = lidar.poll()
        point_cloud = readings_data["pointCloud"]
        all_point_clouds.append(point_cloud)
        frame_count += 1
        print(f"Collected frame {frame_count}: {len(point_cloud)} points")

    # Save collected point clouds ONCE at the end
    output_path = os.path.join(os.path.dirname(__file__), "lidar_data.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(all_point_clouds, f)
    print(f"Saved {len(all_point_clouds)} point clouds to {output_path}")


if __name__ == "__main__":
    collect_lidar_data()