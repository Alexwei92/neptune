import csv
import numpy as np

import matplotlib.pyplot as plt

filename = 'airsim'

data = np.genfromtxt(filename, delimiter=',', skip_header=True)
timestamp = data[:,0]
pos_x = data[:,1]
pos_y = data[:,2]

plt.plot(pos_x, pos_y)
plt.xticks([])
plt.yticks([])
plt.axis('image')
plt.show()