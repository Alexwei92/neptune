import setup_path
import numpy as np
import pandas
import os
import glob
import cv2
import time
import matplotlib.pyplot as plt

from utils import plot_with_cmd_compare
from feature_extract import *
from imitation_learning import exponential_decay

if __name__ == '__main__':
    folder_path = setup_path.parent_dir + '/my_datasets/peng/river/iter0/2021_Feb_09_23_01_33'
    weight_path = setup_path.parent_dir + '/my_outputs/peng/river/iter0/reg_weight.csv'

    # load weight
    weight = np.genfromtxt(weight_path, delimiter=',')

    # visual feature
    feature_path = os.path.join(folder_path, 'feature_preload.pkl')
    X = pandas.read_pickle(feature_path).to_numpy()

    # telemetry feature
    telemetry_data = pandas.read_csv(os.path.join(folder_path, 'airsim.csv'))
    timestamp = telemetry_data['timestamp'][:-1].to_numpy(float)
    timestamp -= timestamp[0]
    timestamp *= 1e-9

    # yaw cmd
    yaw_cmd = telemetry_data['yaw_cmd'][:-1].to_numpy()

    # previous commands with time decaying
    num_prvs = 5
    y_prvs = np.zeros((len(yaw_cmd), num_prvs))
    # prvs_index = exponential_decay(num_prvs)
    prvs_index = [i for i in reversed(range(1, num_prvs+1))]
    for i in range(len(yaw_cmd)):
        for j in range(num_prvs):
            y_prvs[i,j] = yaw_cmd[max(i-prvs_index[j], 0)]

    # yaw rate
    yawRate = np.reshape(telemetry_data['yaw_rate'][:-1].to_numpy(), (-1,1))
    
    # total feature
    X_extra = np.concatenate((y_prvs, yawRate), axis=1)
    X = np.column_stack((X, X_extra))

    # display
    file_list_color = glob.glob(os.path.join(folder_path, 'color', '*.png'))
    file_list_color.sort()

    # dict
    feature_agent = FeatureExtract(feature_config, (480,640))
    time_dict = {}   
    effect = {} 
    bar_name = []
    total_size = 0
    for name, size, _, _ in feature_agent.feature_list:
        bar_name.append(name)
        time_dict[name] = (total_size + size * len(feature_agent.H_points) * len(feature_agent.W_points))
        total_size += size * len(feature_agent.H_points) * len(feature_agent.W_points)


    # plot
    fig_bar, ax_bar = plt.subplots()
    effect_percent = np.zeros((len(time_dict)+3,))
    bar_name.append('cmd_prvs')
    bar_name.append('yawRate')
    bar_name.append('total')
    bar_handle = ax_bar.bar(bar_name, effect_percent, width=0.8, color=['b','g','r','c','m','y','k',(0.5,0.5,0.5)])
    ax_bar.set_ylim((-0.5,0.5))

    i = 0
    tic = time.perf_counter()
    for color_file in file_list_color:
        image = cv2.imread(color_file, cv2.IMREAD_UNCHANGED)
 
        key = cv2.waitKey(1) & 0xFF
        if (key == 27 or key == ord('q')):
            break

        y_pred = np.dot(weight[1:], X[i,:])
        y_pred += weight[0]
        
        for name, size, _, _ in feature_agent.feature_list:
            index = time_dict[name]
            all_size = size * len(feature_agent.H_points) * len(feature_agent.W_points)
            effect[name] = np.dot(weight[index-all_size+1:index+1], X[i, index-all_size:index])

        effect['cmd_prvs'] = np.dot(weight[total_size+1:total_size+5+1], X[i, total_size:total_size+5])
        effect['yawRate'] = np.dot(weight[total_size+5+1:total_size+6+1], X[i, total_size+5:total_size+6])

        k = 0
        for name in effect:
            effect_percent[k] = effect[name] / (y_pred-weight[0])
            # print('{:s}: {:.2%},'.format(name, effect_percent), end=" ")  
            bar_handle[k].set_height(effect[name])      
            k += 1 
        bar_handle[len(bar_handle)-1].set_height(y_pred)  
            
        y_pred = np.clip(y_pred, -1.0, 1.0)
        
        # plot
        plot_with_cmd_compare('reg', image, yaw_cmd[i], y_pred)
        # ax_bar.relim()
        # ax_bar.autoscale_view()
        plt.pause(1e-5)

        elapsed_time = time.perf_counter() - tic
        time.sleep(max(timestamp[i] - elapsed_time, 0))
        i += 1
        # print('\n')