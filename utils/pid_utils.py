import time

from utils import SecondOrderLowPass

class PIDController():
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, scale=1.0, cutoff_freq=5.0, sample_freq=10.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.scale = scale
        self.prev_time = time.perf_counter()
        self.previous_error = 0.0
        self.previous_set = False
        self.sum = 0.0
        self.lowpassfilter = SecondOrderLowPass(cutoff_freq, sample_freq)

    def update(self, target_point, current_point):
        time_now = time.perf_counter()
        dt = time_now - self.prev_time

        error = self.lowpassfilter.update(target_point - current_point)
        if not self.previous_set:
            self.previous_set = True
            self.previous_error = error
            self.prev_time = time_now
            return 0.0
        
        proportional_gain = 0
        derivative_gain = 0
        integral_gain = 0
        if self.kp != 0:
            proportional_gain = error * self.kp

        if self.kd != 0:
            derivative = (error - self.previous_error) / dt
            derivative_gain = derivative * self.kd
        
        if self.ki != 0:
            self.sum += error * dt
            integral_gain = self.sum * self.ki

        self.previous_error = error
        self.prev_time = time_now

        return self.scale * (proportional_gain + derivative_gain + integral_gain)

    def reset(self):
        self.prev_time = time.perf_counter()
        self.previous_error = 0.0
        self.previous_set = False
        self.sum = 0.0
        self.lowpassfilter.reset()