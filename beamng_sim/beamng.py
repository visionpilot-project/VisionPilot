from beamng_sim.lane_detection import process_frame
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera
import cv2
import numpy as np
import time
import math
from scipy import signal


def yaw_to_quat(yaw_deg):
    yaw = math.radians(yaw_deg)
    w = math.cos(yaw / 2)
    z = math.sin(yaw / 2)
    return (0.0, 0.0, z, w)
    
def apply_butterworth_filter(value, buffer, b, a, max_size=10):
    buffer.insert(0, value)
    if len(buffer) > max_size:
        buffer.pop()
    
    if len(buffer) >= 3:
        filtered_value = signal.filtfilt(b, a, buffer)[0]
        return filtered_value
    else:
        return value

# PID Controller Setup
class PIDController:
    def __init__(self, Kp, Ki, Kd, integral_limit=1):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_derivative = 0.0
        self.integral_limit = integral_limit

    def update(self, error, dt):
        if dt <= 0:
            return 0.0
        self.integral += error * dt
        
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)

        derivative = (error - self.previous_error) / dt

        # smooth derivative to reduce noise sensitivity
        derivative = 0.9 * self.previous_derivative + 0.1 * derivative
        self.previous_derivative = derivative

        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error
        return output


beamng = BeamNGpy('localhost', 64256, home=r'C:\Users\user\Documents\beamng-tech\BeamNG.tech.v0.36.4.0')
beamng.open()

scenario = Scenario('west_coast_usa', 'lane_detection')
vehicle = Vehicle('ego_vehicle', model='etk800', licence='PYTHON')

rot = yaw_to_quat(-133.506 + 180)
scenario.add_vehicle(vehicle, pos=(-730.212, 94.630, 118.517), rot_quat=rot)
scenario.make(beamng)
beamng.scenario.load(scenario)
beamng.scenario.start()

camera = Camera(
    'front_cam',
    beamng,
    vehicle,
    requested_update_time=0.01,
    is_using_shared_memory=True,
    pos=(0, -1.3, 1.4),
    dir=(0, -1, 0),
    field_of_view_y=90,
    near_far_planes=(0.1, 1000),
    resolution=(640, 360),
    is_streaming=True,
    is_render_colours=True,
)

# PID controller Parameters + smoothing
pid = PIDController(Kp=0.2, Ki=0.04, Kd=0.05, integral_limit=5.0)

last_lane_center = None
smooth_deviation = 0.0
previous_steering = 0.0
last_time = time.time()

# Butterworth filter setup
butter_order = 3
butter_cutoff = 0.1
butter_b, butter_a = signal.butter(butter_order, butter_cutoff)
deviation_buffer = []
deviation_buffer_max_size = 10

alpha = 0.1 # smoothing factor for deviation (0-1). smaller = smoother/more lag
max_delta = 0.03 # max steering change per loop
base_throttle = 0.1 # baseline throttle

try:
    for step_i in range(1000):
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        beamng.control.step(10)
        images = camera.stream()
        img = np.array(images['colour'])
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        result, metrics = process_frame(img_bgr)

        lane_center = metrics.get('lane_center', None)
        raw_deviation = metrics.get('deviation', None)

        if raw_deviation is not None:
            deviation = raw_deviation
        else:
            deviation = 0.0

        if raw_deviation is None or 'error' in metrics or (isinstance(raw_deviation, (int, float)) and abs(raw_deviation) > 2.0):
            if last_lane_center is not None:
                if raw_deviation is None:
                    deviation = last_lane_center
                else:
                    deviation = 0.8 * raw_deviation + 0.2 * last_lane_center
                lane_center = last_lane_center
            else:
                deviation = 0.0
        else:
            deviation = raw_deviation
            last_lane_center = lane_center

        filtered_deviation = apply_butterworth_filter(deviation, deviation_buffer, butter_b, butter_a, deviation_buffer_max_size)
        
        smooth_deviation = alpha * filtered_deviation + (1.0 - alpha) * smooth_deviation

        deadband = 0.04
        if abs(smooth_deviation) < deadband:
            smooth_deviation = 0.0

        max_jump = 0.5
        deviation_change = deviation - smooth_deviation
        if abs(deviation_change) > max_jump:
            deviation = smooth_deviation

        steering = pid.update(-smooth_deviation, dt)
        steering = float(np.clip(steering, -1.0, 1.0))

        steering = float(np.clip(steering, previous_steering - max_delta, previous_steering + max_delta))
        previous_steering = steering

        throttle = base_throttle * (1.0 - 0.5 * abs(steering) - 0.2 * abs(smooth_deviation))
        throttle = float(np.clip(throttle, 0.03, 0.25))

        if abs(steering) > 0.95:
            throttle = 0.03
            brake = 0.15
        else:
            brake = 0.0 if abs(steering) < 0.9 else 0.08

        vehicle.control(steering=steering, throttle=throttle, brake=brake)

        cv2.imshow('Lane Detection', result)
        if step_i % 30 == 0:
            print(f"[{step_i}] Deviation(raw/smoothed): {deviation:.3f} / {smooth_deviation:.3f} m | Steering: {steering:.3f} | Throttle: {throttle:.3f}")
            print(f"    Curvature: L={metrics.get('left_curverad',0):.1f}m, R={metrics.get('right_curverad',0):.1f}m")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cv2.destroyAllWindows()
    beamng.close()
