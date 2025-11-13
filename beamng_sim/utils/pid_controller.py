import numpy as np


class PIDController:
    def __init__(self, Kp=0.015, Ki=0.0, Kd=0.025, derivative_filter_alpha=0.3, integral_limit=1.0, feedforward=0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_limit = integral_limit
        self.derivative_filter_alpha = derivative_filter_alpha 
        self.feedforward = feedforward
        self.previous_error = 0.0
        self.integral = 0.0
        self.filtered_derivative = 0.0
        
    def update(self, error, dt, feedforward=None):
        if dt <= 0:
            dt = 0.01

        if np.sign(error) != np.sign(self.previous_error) and abs(self.previous_error) > 1e-6:
            self.integral = 0.0

        p_term = self.Kp * error

        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.Ki * self.integral

        raw_derivative = (error - self.previous_error) / dt
        self.filtered_derivative = (
            self.derivative_filter_alpha * raw_derivative + 
            (1 - self.derivative_filter_alpha) * self.filtered_derivative
        )
        d_term = self.Kd * self.filtered_derivative

        self.previous_error = error

        ff = self.feedforward if feedforward is None else feedforward
        output = p_term + i_term + d_term + ff
        return output
    
    def reset(self):
        self.previous_error = 0.0
        self.integral = 0.0
        self.filtered_derivative = 0.0
