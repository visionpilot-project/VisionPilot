from beamng_sim.lane_detection import process_frame
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera
import cv2
import numpy as np

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.previous_error = 0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt if dt > 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error
        return output


beamng = BeamNGpy('localhost', 64256, home=r'D:\beamng-tech\BeamNG.tech.v0.36.4.0')
beamng.open()

scenario = Scenario('west_coast_usa', 'lane_detection')
vehicle = Vehicle('ego_vehicle', model='etk800', licence='PYTHON')
scenario.add_vehicle(vehicle, pos=(-717, 101, 118), rot_quat=(0, 0, 0, 1))
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
    is_render_annotations=False,
    is_render_instance=False,
    is_render_depth=False,
)

pid = PIDController(Kp=0.5, Ki=0.05, Kd=0.1)
dt = 0.1

for _ in range(1000):
    beamng.control.step(10)
    images = camera.stream()
    img = np.array(images['colour'])
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    result, metrics = process_frame(img_bgr)

    lane_center = metrics['lane_center']
    deviation = metrics['deviation']  # meters, positive = left, negative = right
    
    if abs(deviation) > 2 or 'error' in metrics:
        if last_lane_center is not None:
            lane_center = last_lane_center
    else:
        last_lane_center = lane_center

    # ADDED PID Controller Proportional Integral Derivative
    steering = pid.update(deviation, dt)

    steering = max(min(steering, 1), -1)

    max_throttle = 1.0
    k = 0.7

    throttle = max_throttle * (1 - k * abs(steering))
    throttle = max(throttle, 0.05)

    vehicle.control(steering=steering, throttle=throttle)

    cv2.imshow('Lane Detection', result)
    
    if _ % 30 == 0:
        print(f"Curvature: L={metrics['left_curverad']:.1f}m, R={metrics['right_curverad']:.1f}m")
        print(f"Deviation: {metrics['deviation']:.2f}m")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cv2.destroyAllWindows()
beamng.close()