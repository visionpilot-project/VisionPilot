def process_frame(radar_sensor, max_distance = 100):
    """
    Gets the latest RADAR PPI image from shared memory.
    Args:
        radar_sensor: BeamNG radar sensor object
    Returns:
        The latest RADAR PPI image from shared memory
    """

    radar_data = radar_sensor.poll()
    points = radar_data.get('points', [])


    filtered_radar = []
    for point in points:
        range = radar_data[0]
        rel_vel = radar_data[1]
        if range is not None and rel_vel is not None:
            if range > max_distance:
                return None
            else:
                filtered_radar.append((range, rel_vel))
    return filtered_radar
