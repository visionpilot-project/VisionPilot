import numpy as np


def lerp_by_speed(speed, v_low, v_high, val_low, val_high):
    """
    Linear interpolation of a value between two speed breakpoints, held constant outside them.
    """
    if speed <= v_low:
        return val_low
    if speed >= v_high:
        return val_high
    t = (speed - v_low) / (v_high - v_low)
    return val_low + t * (val_high - val_low)


class PIDController:
    """
    PID with clamped output and conditional-integration anti-windup:
    the integral does not grow while the output is saturated and the error pushes further into the limit.
    """

    def __init__(self, kp, ki=0.0, kd=0.0, out_min=-1.0, out_max=1.0, i_max=None, debug=False):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min = out_min
        self.out_max = out_max
        self.i_max = i_max if i_max is not None else (out_max - out_min)
        self.integral = 0.0
        self.prev_error = None
        self.debug = debug

    def update(self, error, dt):
        if dt <= 0:
            dt = 0.01

        p_term = self.kp * error

        # no D on the first update, avoids derivative kick
        if self.prev_error is None:
            d_term = 0.0
        else:
            d_term = self.kd * (error - self.prev_error) / dt
        self.prev_error = error

        unsat = p_term + d_term + self.ki * self.integral
        push_high = unsat >= self.out_max and error > 0
        push_low = unsat <= self.out_min and error < 0
        if not (push_high or push_low):
            self.integral += error * dt
            self.integral = np.clip(self.integral, -self.i_max, self.i_max)
        i_term = self.ki * self.integral

        output = float(np.clip(p_term + i_term + d_term, self.out_min, self.out_max))

        if self.debug:
            print(f"[PID] error={error:.4f}, dt={dt:.4f}, P={p_term:.4f}, I={i_term:.4f}, D={d_term:.4f}, out={output:.4f}")

        return output

    def reset(self):
        self.prev_error = None
        self.integral = 0.0


class LaneKeepingController:
    """
    Steering from the lane deviation at a lookahead distance:
    gain scheduling -> hold on lane loss -> dead-zone -> PID -> low-pass filter -> slew limit.

    Sign: deviation > 0 (vehicle right of lane center) gives steering < 0 (left).
    """

    def __init__(self, cfg, debug=False):
        self.cfg = cfg
        self.pid = PIDController(cfg['kp_low'], cfg['ki'], cfg['kd_low'], out_min=-1.0, out_max=1.0, i_max=cfg['i_max'], debug=debug)
        self.prev_steering = 0.0
        self.last_valid_error = 0.0
        self.invalid_ticks = 0

    def _by_speed(self, speed_kph, key):
        return lerp_by_speed(speed_kph, self.cfg['v_low_kph'], self.cfg['v_high_kph'], self.cfg[f'{key}_low'], self.cfg[f'{key}_high'])

    def lookahead_m(self, speed_kph):
        """Lookahead distance from the camera for the current speed."""
        return self._by_speed(speed_kph, 'lookahead_m')

    def update(self, deviation_ahead, speed_kph, dt):
        """
        Args:
            deviation_ahead: Deviation at the lookahead distance in meters, None if the lane is not detected
            speed_kph: Vehicle speed
            dt: Time step in seconds
        Returns:
            float: Steering between -1 and 1
        """
        cfg = self.cfg

        self.pid.kp = self._by_speed(speed_kph, 'kp')
        self.pid.kd = self._by_speed(speed_kph, 'kd')
        slew_rate = self._by_speed(speed_kph, 'slew_per_s')

        # On short detection drops keep steering on the last error, on longer ones decay it to 0
        if deviation_ahead is not None:
            error = -deviation_ahead
            self.last_valid_error = error
            self.invalid_ticks = 0
        else:
            self.invalid_ticks += 1
            if self.invalid_ticks > cfg['inertia_ticks']:
                self.last_valid_error *= cfg['inertia_decay']
            error = self.last_valid_error

        if abs(error) < cfg['deadzone_m']:
            error = 0.0

        raw = self.pid.update(error, dt)

        # Low-pass filter
        alpha = dt / (cfg['lpf_tau_s'] + dt)
        steering = alpha * raw + (1.0 - alpha) * self.prev_steering

        # Slew rate limit
        max_delta = slew_rate * dt
        steering = self.prev_steering + np.clip(steering - self.prev_steering, -max_delta, max_delta)

        steering = float(np.clip(steering, -1.0, 1.0))
        self.prev_steering = steering
        return steering

    def reset(self):
        self.pid.reset()
        self.prev_steering = 0.0
        self.last_valid_error = 0.0
        self.invalid_ticks = 0
