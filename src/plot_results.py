import csv
import numpy as np
import os
import pandas
import matplotlib.pyplot as plt


def plot_start_point(pos_x, pos_y):
    plt.scatter(pos_x, pos_y, s=100, color='b', marker='*')

def plot_crash_point(pos_x, pos_y):
    plt.scatter(pos_x, pos_y, s=100, color='r', marker='x')

def plot_one_trail(pos_x, pos_y, is_crash, color='b'):
    plt.scatter(pos_x, pos_y, s=1, c=color)
    plot_start_point(pos_x[0], pos_y[0])
    if is_crash:
        plot_crash_point(pos_x[-1], pos_y[-1])

if __name__ == '__main__':

    folder_path = '/home/lab/Documents/Peng/neptune/my_datasets/peng/river/iter1_latent_hg_disp'
    total_N = 0
    for subfolder in os.listdir(folder_path):
        file_path = os.path.join(folder_path, subfolder, 'airsim.csv')
        # data = np.genfromtxt(file_path, delimiter=',', skip_header=True)
        data = pandas.read_csv(file_path)

        if data.iloc[-1,0] == 'crashed':
            crashed = True
        else:
            crashed = False
 
        pos_x = data['pos_x'][:-1].to_numpy()
        pos_y = data['pos_y'][:-1].to_numpy()
        # cmd = data['yaw_cmd'][:-1].to_numpy()

        if subfolder == '2021_Feb_04_07_20_01':
            plot_one_trail(pos_x, pos_y, crashed, 'k')        
        else:
            plot_one_trail(pos_x, pos_y, crashed, 'b')        
            total_N += len(pos_x)
    
    print('Total data points: ',  total_N)
    # plt.xticks([])
    # plt.yticks([])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('image')
    
    plt.show()