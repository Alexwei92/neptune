import setup_path 
import airsim
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter,filtfilt,lfilter

class PIDController():
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_time = time.perf_counter()
        self.previous_error = 0.0
        self.previous_set = False
        self.sum = 0.0
        self.set_point = 0.0

    def update(self, target_point, current_point):
        time_now = time.perf_counter()
        dt = time_now - self.prev_time
        # dt = np.amax([dt, 0.01])

        error = target_point - current_point
        if not self.previous_set:
            self.previous_set = True
            self.previous_error = error
            return 0
        
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

        return proportional_gain + derivative_gain + integral_gain

    def reset_integral(self):
        self.sum = 0.0

class Rangefinder():
    def __init__(self):
        self.N = 10
        self.data = np.array([])

    def update(self, data):
        if len(self.data) < self.N:
            self.data = np.append(self.data, data)
        else:
            self.data = np.append(np.delete(self.data, 0), data)

        return np.mean(self.data)

def butter_lowpass_filter(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_filter(data):
    b, a = butter_lowpass_filter(cutoff=2, fs=10, order=5)
    y = lfilter(b, a, data)
    return y



client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

client.armDisarm(True)
client.enableApiControl(False)



rf = Rangefinder()

true_height = []
filter_height = []
rangefinder_height = []
rf_height = []
i = 0


while True:
    start_time = time.perf_counter()
    data = client.getDistanceSensorData().distance
    # line1.set_ydata(np.append(line1.get_ydata(), rf.update(data)))
    # line1.set_xdata(np.append(line1.get_xdata(), i))
    rangefinder_height.append(data)
    rf_height.append(rf.update(data))

    height = -client.getMultirotorState().kinematics_estimated.position.z_val 
    # line2.set_ydata(np.append(line2.get_ydata(), height))
    # line2.set_xdata(np.append(line2.get_xdata(), i))
    # i += 1
    # ax.relim()
    # ax.autoscale_view()
    # plt.pause(1e-5)
    # time.sleep(0.1)
    true_height.append(height)

    elapsed_time = time.perf_counter() - start_time
    if (1./10 - elapsed_time) < 0.0:
        print('Too fast')
    else:
        time.sleep(1./10 - elapsed_time)

    i += 1

    if i > 500:
        break

plt.plot(true_height, label='true_height')
# plt.plot(rangefinder_height, label='rangefinder_height')
plt.plot(rf_height, label='rf_height')
y_filtered = apply_filter(rangefinder_height)
plt.plot(y_filtered, label='y_filtered')
plt.legend()
plt.show()