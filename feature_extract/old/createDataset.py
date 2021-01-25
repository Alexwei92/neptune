import numpy as np
import cv2
import os
import pandas ÷as pd

from colorFeatures import colorFeatures
from colorFeatures import opticalFlow
from depthFeatures import depthFeatures
import config as cf

# Law's filters
L5 = np.array([.1,.4,.6,.4,.1])
E5 = np.array([-1,-2,0,2,1])
S5 = np.array([-1,0,2,0,-1])
W5 = np.array([-1,2,0,-2,1])
R5 = np.array([1,-4,6,-4,1])

# Laws' masks
# A dictionary of law masks
# The most successful masks are {L5E5, E5S5, R5R5, L5S5, E5L5, S5E5, S5L5}
maskDict = {
"L5L5" : (L5.reshape(5,1) * L5)/25,
"L5E5" : (L5.reshape(5,1) * E5)/25,
"L5S5" : (L5.reshape(5,1) * S5)/25,
"E5L5" : (E5.reshape(5,1) * L5)/25,
"E5E5" : (E5.reshape(5,1) * E5)/25,
"E5S5" : (E5.reshape(5,1) * S5)/25,
"S5L5" : (S5.reshape(5,1) * L5)/25,
"S5E5" : (S5.reshape(5,1) * E5)/25,
"S5S5" : (S5.reshape(5,1) * S5)/25
}

law_L5L5 = cv2.cuda.createLinearFilter(cv2.CV_32F, cv2.CV_32F, kernel=maskDict["L5L5"].astype(np.float32))
law_L5E5 = cv2.cuda.createLinearFilter(cv2.CV_32F, cv2.CV_32F, kernel=maskDict["L5E5"].astype(np.float32))
law_L5S5 = cv2.cuda.createLinearFilter(cv2.CV_32F, cv2.CV_32F, kernel=maskDict["L5S5"].astype(np.float32))
law_E5E5 = cv2.cuda.createLinearFilter(cv2.CV_32F, cv2.CV_32F, kernel=maskDict["E5E5"].astype(np.float32))
law_E5S5 = cv2.cuda.createLinearFilter(cv2.CV_32F, cv2.CV_32F, kernel=maskDict["E5S5"].astype(np.float32))
law_S5S5 = cv2.cuda.createLinearFilter(cv2.CV_32F, cv2.CV_32F, kernel=maskDict["S5S5"].astype(np.float32))

law_masks = [law_L5L5, law_L5E5, law_L5S5, law_E5E5, law_E5S5, law_S5S5]


if __name__ == '__main__':
    folder_name = '2020_Jul_27_16_53_00/'
    filename = 'frame000002.jpg'
    
    # Telemtetry data
    dataset = pd.read_csv(folder_name + '_my_telemetry.csv', skiprows=[1])

    frame_num = 2
    features = np.zeros((cf.COLOR_FEATURES_SIZE+cf.DEPTH_FEATURES_SIZE+cf.FLOW_FEATURES, len(dataset)))
    old_color = cv2.imread(folder_name + 'color/' + 'frame000001.jpg', cv2.IMREAD_UNCHANGED)

    while(os.path.isfile(folder_name + 'color/' + filename) and os.path.isfile(folder_name + 'depth/' + filename)):
        # for test purposes
        print(filename)

        # Import images
        color_image = cv2.imread(folder_name + 'color/' + filename, cv2.IMREAD_UNCHANGED)
        depth_image = cv2.imread(folder_name + 'depth/' + filename, cv2.IMREAD_GRAYSCALE)
        
        # Process images
        features[0:cf.COLOR_FEATURES_SIZE, (frame_num - 2)] = colorFeatures(color_image, law_masks)
        features[cf.COLOR_FEATURES_SIZE:(cf.COLOR_FEATURES_SIZE+cf.DEPTH_FEATURES_SIZE), (frame_num - 2)] = depthFeatures(depth_image)
        features[(cf.COLOR_FEATURES_SIZE+cf.DEPTH_FEATURES_SIZE):(cf.COLOR_FEATURES_SIZE+cf.DEPTH_FEATURES_SIZE+cf.FLOW_FEATURES), (frame_num - 2)] = opticalFlow(old_color, color_image)

        frame_num = frame_num + 1
        filename = 'frame' + '%06d' % frame_num + '.jpg'
        old_color = color_image.copy()

    # Create column names for the image features
    image_columns = []

    # Color feature names
    for i in range(cf.COLOR_COLS):
        for j in range(cf.COLOR_ROWS):
            w_index = i + j*cf.COLOR_COLS
            for k in range(cf.RADON_ANGLES*2):
                image_columns = image_columns + [('%d' % w_index + 'r' + '%d' % k)]
            for k in range(cf.BINS):
                image_columns = image_columns + [('%d' % w_index + 's' + '%d' % k)]
            for k in range(cf.NUM_LAWS):
                image_columns = image_columns + [('%d' % w_index + 'l' + '%d' % k)]
    
    # Depth feature names
    for i in range(cf.DEPTH_COLS):
        for j in range(cf.DEPTH_ROWS):
            w_index = i + j*cf.DEPTH_COLS
            image_columns = image_columns + [('%d' % w_index + 'd')]

    # Optical Flow feature names
    for i in range(cf.COLOR_COLS):
        for j in range(cf.COLOR_ROWS):
            w_index = i + j*cf.COLOR_COLS
            for k in range(cf.NUM_FLOWS):
                image_columns = image_columns + [('%d' % w_index + 'f' + '%d' % k)]
    for k in range(cf.NUM_FLOWS):
                image_columns = image_columns + [('ff' + '%d' % k)]

    dataset = pd.concat([dataset, pd.DataFrame(features.transpose(), index= dataset.index, columns= image_columns)], axis=1)
    
    print(dataset)