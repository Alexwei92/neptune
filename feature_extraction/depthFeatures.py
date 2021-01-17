import numpy as np
#import cv2

from colorFeatures import sliding_window
import config as cf

def depthFeatures(image):
    image = 255 - image

    image_height, image_width = image.shape

    # Split the raw image into windows
    split_size = (cf.DEPTH_ROWS, cf.DEPTH_COLS) # split size e.g., (#height, #width) (rows, columns)

    H_points, H_size = sliding_window(image_height, split_size[0], cf.DEPTH_OVERLAP, flag=1) # flag=0 if provide the exact size, flag=1 if provide the number of windows
    W_points, W_size = sliding_window(image_width, split_size[1], cf.DEPTH_OVERLAP, flag=1)

    # Get features for each window.
    features = np.zeros(cf.DEPTH_FEATURES_SIZE)
    k = 0
    for i in H_points:
        for j in W_points:
            cropped_image = image[i:i+H_size, j:j+W_size]
            #TODO: Currently just the average of all pixels
            features[k] = np.mean(np.abs(cropped_image))
            k += 1

    return features
