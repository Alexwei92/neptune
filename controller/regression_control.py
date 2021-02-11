import numpy as np
import os

from feature_extract import *
from imitation_learning import exponential_decay

class RegCtrl():
    '''
    Linear Regression Controller
    '''
    def __init__(self, num_prvs, image_size, weight_file_path, printout=False):
        self.load_weight_from_file(weight_file_path)
        self.feature_agent = FeatureExtract(feature_config, image_size, printout)
        self.num_prvs = num_prvs
        # self.prvs_index = exponential_decay(num_prvs)
        self.prvs_index = [5,4,3,2,1]
        self.cmd_prvs = np.zeros((max(self.prvs_index),))
        print('The linear regression controller is initialized.')

    def load_weight_from_file(self, file_path):
        self.weight = np.genfromtxt(file_path, delimiter=',')
        print('Load weight from: ', file_path)

    def predict(self, image_color, image_depth, yawRate):
        X = self.feature_agent.step(image_color, image_depth)
        for index in self.prvs_index:
            X = np.append(X, self.cmd_prvs[index-1])
        X = np.append(X, yawRate)
        y = np.dot(self.weight[1:], X)
        y += self.weight[0]
        
        # if np.abs(y) < 1e-3:
        #     y = 0.0

        # TODO: Normalize and restrict the output
        if y > 1.0:
            y = 1.0
        if y < -1.0:
            y = -1.0

        for i in reversed(range(1,len(self.cmd_prvs))):
            self.cmd_prvs[i] = self.cmd_prvs[i-1]
        self.cmd_prvs[0] = y

        return y

    def reset_prvs(self):
        self.cmd_prvs.fill(0.0)