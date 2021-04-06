import setup_path
import numpy as np
import pandas
import os
import glob
import cv2
import time
import matplotlib.pyplot as plt

from utils import plot_with_cmd
from feature_extract import *

if __name__ == '__main__':
    folder_path = '/media/lab/Hard Disk/my_datasets/subject5/map2/iter0/2021_Feb_22_15_15_27'
    
    # telemetry feature
    telemetry_data = pandas.read_csv(os.path.join(folder_path, 'airsim.csv'))
    timestamp = telemetry_data['timestamp'][:-1].to_numpy(float)
    timestamp -= timestamp[0]
    timestamp *= 1e-9 / 4

    # yaw cmd
    yaw_cmd = telemetry_data['yaw_cmd'][:-1].to_numpy()

    # display
    file_list_color = glob.glob(os.path.join(folder_path, 'color', '*.png'))
    file_list_color.sort()

    i = 0
    cv2.namedWindow('disp')
    time.sleep(10.0)
    
    tic = time.perf_counter()
    for color_file in file_list_color:
        image = cv2.imread(color_file, cv2.IMREAD_UNCHANGED)
 
        key = cv2.waitKey(1) & 0xFF
        if (key == 27 or key == ord('q')):
            break

        # plot
        plot_with_cmd('disp', image, yaw_cmd[i], True)

        plt.pause(1e-5)

        elapsed_time = time.perf_counter() - tic
        time.sleep(max(timestamp[i] - elapsed_time, 0))
        i += 1