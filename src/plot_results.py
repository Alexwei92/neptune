import csv
import numpy as np
import os
import matplotlib.pyplot as plt


def plot_start_point(pos_x, pos_y):
    plt.scatter(pos_x, pos_y, s=100, color='r', marker='*')

def plot_crash_point(pos_x, pos_y):
    plt.scatter(pos_x, pos_y, s=100, color='r', marker='x')

def plot_one_trail(pos_x, pos_y, is_crash, color='b'):
    plt.scatter(pos_x, pos_y, s=2, c=color)
    plot_start_point(pos_x[0], pos_y[0])
    if is_crash:
        plot_crash_point(pos_x[-1], pos_y[-1])

if __name__ == '__main__':

    folder_path = '/home/lab/Documents/Peng/neptune/my_datasets/peng/river/iter2'
    total_N = 0
    for subfolder in os.listdir(folder_path):
        file_path = os.path.join(folder_path, subfolder, 'airsim.csv')
        data = np.genfromtxt(file_path, delimiter=',', skip_header=True)
        N = 0
        crash_count = 0
        while True:
            tmp = (data[:,-2] == crash_count)
            # if crash_count > 0:
            #     plot_one_trail(data[tmp,1], data[tmp,2], True, 'b')
            # else:
            #     plot_one_trail(data[tmp,1], data[tmp,2], False, 'k')
            if subfolder == '2021_Feb_04_07_17_29':
                plot_one_trail(data[tmp,1], data[tmp,2], False, 'k')
            else:
                plot_one_trail(data[tmp,1], data[tmp,2], True, 'b')
            
            N += np.sum(tmp)
            if N == len(data):
                break    
            
            crash_count += 1
        
        total_N += N
    
    print('Total data points: ',  total_N)
    # plt.xticks([])
    # plt.yticks([])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('image')
    
    plt.show()