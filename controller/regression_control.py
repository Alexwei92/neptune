import numpy as np
import os
import pickle
import math

from feature_extract import *
from imitation_learning import exponential_decay

class RegCtrl():
    '''
    Linear Regression Controller
    '''
    def __init__(self, num_prvs, prvs_mode, image_size, weight_file_path, printout=False):
        self.load_weight_from_file(weight_file_path)
        self.feature_agent = FeatureExtract(feature_config, image_size, printout)
        self.num_prvs = num_prvs
        if num_prvs > 0:
            if prvs_mode == 'exponential':
                self.prvs_index = exponential_decay(num_prvs)
            elif prvs_mode == 'linear':
                self.prvs_index = [i for i in reversed(range(1, num_prvs+1))]
        print('The linear regression controller is initialized.')

    def load_weight_from_file(self, file_path):
        self.file_type = file_path[-3:]
        if self.file_type == 'csv':
            self.weight = np.genfromtxt(file_path, delimiter=',')
        elif self.file_type == 'pkl':
            self.model = pickle.load(open(file_path, 'rb'))['Model']
        print('Load regression weight from {:s}.'.format(file_path))

    def predict(self, image_color, image_depth, yawRate, cmd_history):
        X = self.feature_agent.step(image_color, image_depth, 'RGB')
        yawRate_norm = yawRate * (180.0 / math.pi) / 45.0
        if self.num_prvs > 0:
            for index in self.prvs_index:
                X = np.append(X, cmd_history[-index])
            X = np.append(X, yawRate_norm)
        else:
            X = yawRate_norm
        
        if self.file_type == 'csv':
            y_pred = np.dot(self.weight[1:], X)
            y_pred += self.weight[0]
        elif self.file_type == 'pkl':
            y_pred, = self.model.predict(np.reshape(X, (1,-1)))
        
        if np.abs(y_pred) < 1e-3:
            y_pred = 0.0  

        return np.clip(y_pred, -1.0, 1.0)