import setup_path 
import airsim
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter,filtfilt,lfilter


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
# fig, ax = plt.subplots()
# line1, = ax.plot([0], [0], label='rangefinder')
# line2, = ax.plot([0], [0], label='true')

# landed = client.getMultirotorState().landed_state
# if landed == airsim.LandedState.Landed:
#     print("taking off...")
#     client.takeoffAsync()
# else:
#     print("already flying...")
#     client.hoverAsync()

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