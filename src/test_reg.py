import setup_path
import glob
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from controller import RegCtrl

if __name__ == '__main__':
    image_size = (480, 640)
    model_path = 'my_outputs/peng/river/iter0'
    folder_path = 'my_datasets/peng/river/iter0/2021_Jan_27_22_13_10'
    reg_weight_path = os.path.join(setup_path.parent_dir, model_path, 'reg_weight.csv')
    
    controller_agent = RegCtrl(image_size, reg_weight_path)

    # print(subfolder_path)
    file_list_color = glob.glob(os.path.join(setup_path.parent_dir, folder_path, 'color', '*.png'))
    file_list_depth = glob.glob(os.path.join(setup_path.parent_dir, folder_path, 'depth', '*.png'))
    file_list_color.sort()
    file_list_depth.sort()

    telemetry_data = np.genfromtxt(os.path.join(setup_path.parent_dir, folder_path, 'airsim.csv'), 
            delimiter=',', skip_header=True, dtype=np.float32)
    y = telemetry_data[:, 6] # yaw_cmd
    # yaw = np.reshape(telemetry_data[:, index-2], (-1,1)) # yaw
    yawRate = np.reshape(telemetry_data[:, 6-1], (-1,1)) # yaw rate
    
    i = 0
    for color_file, depth_file in zip(file_list_color, file_list_depth):
        image_color = cv2.imread(color_file, cv2.IMREAD_UNCHANGED)
        image_depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        y_pred = controller_agent.predict(image_color, image_depth, yawRate[i])
        print(y_pred, y[i])
        i += 1

    # img = np.zeros(image_size, dtype=np.uint8)
    # img = 255 - img

    # cv2.imshow('test', img)
    # cv2.waitKey(0)




