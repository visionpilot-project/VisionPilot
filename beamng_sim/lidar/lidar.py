import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def collect_lidar_data(beamng, lidar_data):
    beamng.control.step(10)
    readings_data = lidar_data
    point_cloud = readings_data["pointCloud"]
    return point_cloud