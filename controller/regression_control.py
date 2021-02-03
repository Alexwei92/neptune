import numpy as np
import os

from feature_extract import *

class RegCtrl():
    '''
    Linear Regression Controller
    '''
    def __init__(self, image_size, weight_file_path):
        self.load_weight_from_file(weight_file_path)
        self.feature_agent = FeatureExtract(feature_config, image_size)
        print('The linear regression controller is initialized.')

    def load_weight_from_file(self, file_path):
        self.weight = np.genfromtxt(file_path, delimiter=',', dtype=np.float32)
        print('Load weight from: ', file_path)

    def predict(self, image_color, image_depth, yawRate):
        X = self.feature_agent.step(image_color, image_depth)
        X = np.append(X, yawRate)
        y = np.dot(self.weight[1:], X)
        y += self.weight[0]

        # TODO: Normalize and restrict the output
        if y > 1.0:
            y = 1.0
        if y < -1.0:
            y = -1.0
        return y