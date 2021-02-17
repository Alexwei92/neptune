import setup_path
import os
import glob
import numpy as np
import cv2
import time

# from feature_extract import *

def test_func(x):
    tmp = x
    tmp += 100
    return tmp


if __name__=="__main__":

    # path = os.path.join(setup_path.parent_dir, 'my_datasets/peng/test/2021_Feb_09_23_01_33')

    # file_list_color = glob.glob(os.path.join(path, 'color', '*.png'))
    # file_list_depth = glob.glob(os.path.join(path, 'depth', '*.png'))
    # file_list_color.sort()
    # file_list_depth.sort()

    # feature_agent = FeatureExtract(feature_config, (480,640))

    # for color_file, depth_file in zip(file_list_color, file_list_depth):
    #     print(color_file)
    #     image_color = cv2.imread(color_file, cv2.IMREAD_UNCHANGED)
    #     image_depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
    #     feature_agent.step(image_color, image_depth)
        # i += 1

    filename = '0000000.png'
    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    img_gpu = cv2.cuda_GpuMat()
    img_gpu.upload(image)

    N = 1
    tic = time.perf_counter()
    for i in range(N):
        img_gpu.upload(image)
    print(time.perf_counter()-tic)

    tic = time.perf_counter()
    for i in range(N):
        image_gpu = cv2.cuda_GpuMat(image)
    print(time.perf_counter()-tic)
    
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # cv2.imshow('disp', image)
    # cv2.waitKey(0)

