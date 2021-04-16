import setup_path
import numpy as np
import pandas
import os
import glob
import cv2
import time
import matplotlib.pyplot as plt

from utils import plot_with_cmd_replay
from feature_extract import *
from imitation_learning import exponential_decay

def read_data(folder_path):
    # telemetry feature
    telemetry_data = pandas.read_csv(os.path.join(folder_path, 'airsim.csv'))
    timestamp = telemetry_data['timestamp'][:-1].to_numpy(float)
    timestamp -= timestamp[0]
    timestamp *= 1e-9

    # yaw cmd
    yaw_cmd = telemetry_data['yaw_cmd'][:-1].to_numpy()

    # flag
    flag = telemetry_data['flag'][:-1].to_numpy(dtype=bool)

    # image
    file_list_color = glob.glob(os.path.join(folder_path, 'color', '*.png'))
    file_list_color.sort()

    return timestamp, yaw_cmd, ~flag, file_list_color

if __name__ == '__main__':
    folder_path = '/media/lab/Hard Disk/my_datasets/test/2021_Apr_12_12_03_42'

    # read data
    timestamp, yaw_cmd, flag, file_list_color = read_data(folder_path)

    # Video handler   
    out = cv2.VideoWriter('end_2_end_depth.avi',cv2.VideoWriter_fourcc(*'XVID'), 10, (640, 480))     

    # Start the loop
    i = 0
    tic = time.perf_counter()
    for color_file in file_list_color:
        image = cv2.imread(color_file, cv2.IMREAD_UNCHANGED)
 
        key = cv2.waitKey(1) & 0xFF
        if (key == 27 or key == ord('q')):
            break
        
        # plot
        image_with_cmd = plot_with_cmd_replay('replay', image, yaw_cmd[i]/2, flag[i])
        out.write(image_with_cmd)
        plt.pause(1e-5)

        elapsed_time = time.perf_counter() - tic
        time.sleep(max(timestamp[i] - elapsed_time, 0))
        i += 1
    
    out.release()