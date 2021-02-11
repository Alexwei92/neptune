import pandas
import numpy as np


# Exponential decaying function
def exp_decay(num_prvs=5, max_prvs=15, ratio=1.5):
    y = []
    for t in range(0,num_prvs):
        y.append(int(np.ceil(max_prvs * np.exp(-t/ratio))))
    return y

filename='airsim.csv'

# read telemetry csv       
telemetry_data = pandas.read_csv(filename)

# Yaw cmd
y = telemetry_data['yaw_cmd'][100:130].to_numpy()

# Previous commands with time decaying
num_prvs = 5
y_prvs = np.zeros((len(y), num_prvs))
prvs_index = exp_decay(num_prvs)

for i in range(len(y)):
    for j in range(num_prvs):
        y_prvs[i,j] = y[max(i-prvs_index[j], 0)]

y = np.reshape(y, (-1,1))
print(np.column_stack([y, y_prvs]))