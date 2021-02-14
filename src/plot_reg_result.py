import setup_path
import numpy as np
import pandas
import os
import glob
import cv2
import time

from utils import plot_with_cmd_compare
from feature_extract import *

if __name__ == '__main__':
    folder_path = '/home/lab/Documents/Peng/neptune/my_datasets/peng/river/iter0/2021_Feb_09_23_01_33'
    weight_path = '/home/lab/Documents/Peng/neptune/my_outputs/peng/river/iter0/reg_weight_test.csv'

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

    # Yaw cmd
    yaw_cmd = telemetry_data['yaw_cmd'][:-1].to_numpy()

    # Previous commands with time decaying
    num_prvs = 5
    y_prvs = np.zeros((len(yaw_cmd), num_prvs))
    prvs_index = [5,4,3,2,1]
    
    for i in range(len(yaw_cmd)):
        for j in range(num_prvs):
            y_prvs[i,j] = yaw_cmd[max(i-prvs_index[j], 0)]

    # Yaw rate
    yawRate = np.reshape(telemetry_data['yaw_rate'][:-1].to_numpy(), (-1,1))
    X_extra = np.concatenate((y_prvs, yawRate), axis=1)

    # total feature
    X = np.column_stack((X, X_extra))

    # disp
    file_list_color = glob.glob(os.path.join(folder_path, 'color', '*.png'))
    file_list_color.sort()

    # Dict
    feature_agent = FeatureExtract(feature_config, (480,640))
    time_dict = {}
    for name, _, _ in feature_agent.feature_list:
        time_dict[name] = []
    time_dict['depth'] = []
    
    total_size = 0
    for i in feature_agent.H_points:
        for j in feature_agent.W_points:
            for name, size, function in feature_agent.feature_list:
                time_dict[name].append(total_size)
                total_size += size

    for i in feature_agent.H_points:
        for j in feature_agent.W_points:      
            time_dict['depth'].append(total_size)      
            total_size += feature_agent.depth_size

    # print(time_dict)

    i = 0
    tic = time.perf_counter()
    for color_file in file_list_color:
        image = cv2.imread(color_file, cv2.IMREAD_UNCHANGED)
 
        key = cv2.waitKey(1) & 0xFF
        if (key == 27 or key == ord('q')):
            break

        y_pred = np.dot(weight[1:], X[i,:])
        y_pred += weight[0]

        effect_hough = 0
        hough_size = 15
        for index in time_dict['hough']:
            effect_hough += np.dot(weight[index+1:index+1+hough_size], X[i,index:index+hough_size])

        effect_tensor = 0
        tensor_size = 15
        for index in time_dict['structure_tensor']:
            effect_tensor += np.dot(weight[index+1:index+1+tensor_size], X[i,index:index+tensor_size])

        effect_law = 0
        law_size = 9
        for index in time_dict['law_mask']:
            effect_law += np.dot(weight[index+1:index+1+law_size], X[i,index:index+law_size])

        effect_flow = 0
        flow_size = 3
        for index in time_dict['optical_flow']:
            effect_flow += np.dot(weight[index+1:index+1+flow_size], X[i,index:index+flow_size])

        effect_depth = 0
        depth_size = 1
        for index in time_dict['depth']:
            effect_depth += np.dot(weight[index+1:index+1+depth_size], X[i,index:index+depth_size])

        effect_cmd_prvs = 0
        cmd_prvs_size = 5
        index = feature_agent.get_size()
        effect_cmd_prvs += np.dot(weight[index+1:index+1+cmd_prvs_size], X[i,index:index+cmd_prvs_size])
        
        effect_yawRate = 0
        index = feature_agent.get_size() + cmd_prvs_size
        effect_yawRate += np.dot(weight[index+1:index+1+1], X[i,index:index+1])  

        # print(effect_hough, y_pred-weight[0]) 
        effect_hough /= (y_pred-weight[0])
        effect_tensor /= (y_pred-weight[0])
        effect_law /= (y_pred-weight[0])
        effect_flow /= (y_pred-weight[0])
        effect_depth /= (y_pred-weight[0])
        effect_cmd_prvs /= (y_pred-weight[0])
        effect_yawRate /= (y_pred-weight[0])

        print('hough: {:.2%}, tensor: {:.2%}, law: {:.2%}, flow: {:.2%}, depth: {:.2%}, cmd_prvs: {:.2%}, yawRate: {:.2%}'\
                        .format(effect_hough, effect_tensor, effect_law, effect_flow, effect_depth, effect_cmd_prvs, effect_yawRate))

        if y_pred > 1.0:
            y_pred = 1.0
        if y_pred < -1.0:
            y_pred = -1.0

        
        plot_with_cmd_compare('reg', image, yaw_cmd[i], y_pred)
        
        elapsed_time = time.perf_counter() - tic
        time.sleep(max(timestamp[i]-elapsed_time, 0))
        i += 1