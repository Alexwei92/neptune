"""
Plot the trajectories 
"""
import csv
import numpy as np
import os
import pandas
import matplotlib.pyplot as plt

# Auxiliary functions
def plot_start_point(axis, pos_x, pos_y):
    axis.scatter(pos_x, pos_y, s=50, color='g', marker='o', alpha=0.5)

def plot_end_point(axis, pos_x, pos_y):
    axis.scatter(pos_x, pos_y, s=25, color='c', facecolors='none', marker='s', alpha=0.5)

def plot_crash_point(axis, pos_x, pos_y):
    axis.scatter(pos_x, pos_y, s=50, color='r', marker='x', alpha=1)

def plot_one_trail(axis, pos_x, pos_y, is_crashed, color='b'):
    axis.plot(pos_x, pos_y, linewidth=1.5, color=color, alpha=0.5)
    plot_start_point(axis, pos_x[0], pos_y[0])
    if is_crashed:
        plot_crash_point(axis, pos_x[-1], pos_y[-1])
    else:
        plot_end_point(axis, pos_x[-1], pos_y[-1])

def read_data(folder_path, axis):
    N = 0
    success_count = 0
    failure_count = 0
    for subfolder in os.listdir(folder_path):
        file_path = os.path.join(folder_path, subfolder, 'airsim.csv')
        data = pandas.read_csv(file_path)

        if data.iloc[-1,0] == 'crashed':
            crashed = True
            failure_count += 1
        else:
            crashed = False
            success_count += 1
 
        pos_x = data['pos_x'][:-1].to_numpy()
        pos_y = data['pos_y'][:-1].to_numpy()

        plot_one_trail(axis, pos_x, pos_y, crashed, 'b')        
        N += len(pos_x)
    return N, success_count, failure_count

if __name__ == '__main__':

    dataset_dir = '/media/lab/Hard Disk/' + 'my_datasets'
    
    # Subject list
    subject_list = [
        'subject1',
        'subject2',
        'subject3',
        'subject4',
        'subject5',
        'subject6',
        'subject7',
        'subject8',
        'subject9',
        'subject10',
    ]
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
    # fig, axes = plt.subplots(2, 5, sharex=True, sharey=True)
    fig, axes = plt.subplots(2, 5, sharex=False, sharey=False)

    # Start the loop
    N_data = [0 for map in map_list]
    counter = [[0, 0] for map in map_list]
    for subject in subject_list:
        for map in os.listdir(os.path.join(dataset_dir, subject)):
            if map in map_list:
                folder_path = os.path.join(dataset_dir, subject, map, iteration)
                index = map_list.index(map)
                N, num_success, num_failure = read_data(folder_path, axes.flat[index])
                N_data[index] += N
                counter[index][0] += num_success
                counter[index][1] += num_failure

    [print('{:s}: {:d} (s={:d}, f={:d})'.format(map_list[i], N_data[i], counter[i][0], counter[i][1])) \
                                         for i in range(len(N_data))] 
    print('Total samples = {:d}'.format(sum(N_data)))
     
    # Success rate
    for axis, map, count in zip(axes.flat, map_list, counter):
        if sum(count) == 0:
            success_rate = 0
        else:
            success_rate = count[0]/sum(count)
        axis.set_title('{:s} ({:.1%})'.format(map, success_rate), fontsize=11)
        
    for axis in axes.flat:  
        # axis.axis('image')
        axis.set_xticks([])
        axis.set_yticks([])
    
    plt.show()