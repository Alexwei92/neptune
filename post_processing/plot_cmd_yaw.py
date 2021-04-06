"""
Plot cmd vs. yaw_rate
"""
import csv
import numpy as np
import os
import pandas
import matplotlib.pyplot as plt
import math

# Auxiliary functions
def read_data(folder_path, axis):
    color_list = ['b', 'r', 'g', 'c']
    index = 0
    for subfolder in os.listdir(folder_path):
        file_path = os.path.join(folder_path, subfolder, 'airsim.csv')
        data = pandas.read_csv(file_path)
 
        timestamp = data['timestamp'][:-1].to_numpy(float)  
        # timestamp -= timestamp[0]   
        timestamp *= 1e-9
        yaw_cmd = data['yaw_cmd'][:-1].to_numpy(dtype=np.float32)
        yaw_rate = data['yaw_rate'][:-1].to_numpy(dtype=np.float32)
        yaw_rate *= (180.0 / math.pi) / 45.0
      
        axis.plot(timestamp, yaw_cmd, '--', linewidth=0.5, color='b')
        axis.plot(timestamp, yaw_rate, '-', linewidth=0.5, color='b')
        index += 1
        return yaw_cmd

if __name__ == '__main__':

    dataset_dir = '/media/lab/Hard Disk/' + 'my_datasets'
    
    # Subject
    subject = 'subject2'

    # Map list
    map_list = [
        'map1',
        'map2',
        'map3',
        'map4',
        'map5',
        'map7',
        'map8',
        'map9',
        'o1',
        'o2',
    ]

    # Iteration
    iteration = 'iter0'

    # Init the plot
    fig, axes = plt.subplots(2, 5, sharex=False, sharey=True)

    # yaw_cmd
    all_cmd = np.empty((0,),dtype=np.float32)

    # Start the loop
    N_data = [0 for map in map_list]
    counter = [[0, 0] for map in map_list]
    for map in os.listdir(os.path.join(dataset_dir, subject)):
        if map in map_list:
            folder_path = os.path.join(dataset_dir, subject, map, iteration)
            index = map_list.index(map)
            yaw_cmd = read_data(folder_path, axes.flat[index])
            all_cmd = np.concatenate((all_cmd, yaw_cmd), axis=0)
        
    for axis, map in zip(axes.flat, map_list):  
        axis.set_yticks([-1,0,1])
        axis.set_xticks([])
        axis.set_title('{:s}'.format(map), fontsize=11)
    
    # Plot the Histogram
    plt.figure()
    plt.hist(all_cmd, bins=30, range=(-1,1), density=True, color='b', alpha=0.8)
    plt.title('yaw_cmd histogram')
   
    plt.show()