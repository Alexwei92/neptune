import numpy as np
import os

from feature_extract import *
from controller import BaseCtrl

class LGCtrl(BaseCtrl):
    '''
    Linear Regression Controller
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        weight_file_path = kwargs.get('weight_file_path')
        image_size = kwargs.get('image_size')
        self.load_weight_from_file(weight_file_path)
        self.feature_agent = FeatureExtract(feature_config, image_size)

    def load_weight_from_file(self, file_path):
        self.weight = np.genfromtxt(file_path, delimiter=',')

    def predict(self, image_color, image_depth):
        X = self.feature_agent.step(image_color, image_depth)
        y = np.dot(self.weight[1:], X)
        y += self.weight[0]
        # TODO: Normalize and restrict the output
        if y > 1.0:
            y = 1.0
        if y < -1.0:
            y = -1.0
        return y

    def step(self, yaw_cmd, flight_mode):
        if flight_mode == 'hover':
            self.send_command(0.0, is_hover=True)
        elif flight_mode == 'mission':
            self.send_command(yaw_cmd, is_hover=False)
        else:
            print('Unknown flight_mode: ', flight_mode)
            raise Exception