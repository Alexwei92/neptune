import glob
import os
import numpy as np
import cv2

from feature_extract import *

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score 
# import statsmodels.api as sm

# Linear Regression
def calculate_regression(X, y, disp_summary=False):
    reg = LinearRegression().fit(X, y)
    R_square = reg.score(X, y)
    print('r_square = {:.6f}'.format(R_square))

    # if disp_summary:
    #     X2 = sm.add_constant(X)
    #     est = sm.OLS(y, X2)
    #     est2 = est.fit()
    #     print(est2.summary())

    return reg.coef_, reg.intercept_, R_square

class RegTrain():
    '''
    Linear Regression Training Agent
    '''
    def __init__(self, folder_path, image_size, cmd_index, preload=False):
        self.feature_agent = FeatureExtract(feature_config, image_size)
        self.cmd_index = cmd_index
        self.X = np.empty((0, self.feature_agent.get_size()+1), dtype=np.float32)
        self.y = np.empty((0,), dtype=np.float32)

        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            print(subfolder_path)
            file_list_color = glob.glob(os.path.join(subfolder_path, 'color', '*.png'))
            file_list_depth = glob.glob(os.path.join(subfolder_path, 'depth', '*.png'))
            file_list_color.sort()
            file_list_depth.sort()
            if len(file_list_color) != len(file_list_depth):
                raise Exception("The size of color and depth images does not match!")
            
            # Visual feature
            X, y = self.get_sample(file_list_color, file_list_depth, subfolder_path, preload)
            X[X==np.inf] = 0
            self.X = np.concatenate((self.X, X), axis=0)
            self.y = np.concatenate((self.y, y), axis=0)

    def get_sample(self, file_list_color, file_list_depth, folder_path, preload):
        default_filepath = os.path.join(folder_path, 'sample_preload.csv')
        
        if preload and os.path.isfile(default_filepath):
            X = np.genfromtxt(default_filepath, delimiter=',', dtype=np.float32)
        else:
            X = np.zeros((len(file_list_color), self.feature_agent.get_size()))
            i = 0
            for color_file, depth_file in zip(file_list_color, file_list_depth):
                print(color_file)
                image_color = cv2.imread(color_file, cv2.IMREAD_UNCHANGED)
                image_depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
                X[i,:] = self.feature_agent.step(image_color, image_depth)
                i += 1
            np.savetxt(default_filepath, X, delimiter=',')
            
        X_extra, y = self.read_telemetry(folder_path, self.cmd_index)
        X = np.concatenate((X, X_extra), axis=1)
        print('Load linear regression samples successfully.')
        return X, y

    def read_telemetry(self, folder_path, index):
        # read telemetry csv
        telemetry_data = np.genfromtxt(os.path.join(folder_path, 'airsim.csv'), 
                    delimiter=',', skip_header=True, dtype=np.float32)
        y = telemetry_data[:, index] # yaw_cmd
        # yaw = np.reshape(telemetry_data[:, index-2], (-1,1)) # yaw
        yawRate = np.reshape(telemetry_data[:, index-1], (-1,1)) # yaw rate
        return yawRate, y

    def calculate_weight(self):
        weight, intercept, r2 = calculate_regression(self.X, self.y)
        self.weight = np.append(intercept, weight)
        self.r2 = r2

    def save_weight(self, file_path):
        np.savetxt(file_path, self.weight, delimiter=',')
        print('Save weight to: ', file_path)

    def train(self):
        self.calculate_weight()
        print('Number of weight values: ', len(self.weight))