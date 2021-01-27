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
    def __init__(self, folder_path, cmd_index, preload=False):
        self.folder_path = folder_path

        self.file_list_color = glob.glob(os.path.join(self.folder_path, 'color', '*.png'))
        self.file_list_depth = glob.glob(os.path.join(self.folder_path, 'depth', '*.png'))
        self.file_list_color.sort()
        self.file_list_depth.sort()
        if len(self.file_list_color) != len(self.file_list_depth):
            raise Exception("The size of color and depth images does not match!")

        # Visual feature
        tmp_img = cv2.imread(self.file_list_color[0], cv2.IMREAD_UNCHANGED)
        self.feature_agent = FeatureExtract(feature_config, tmp_img.shape[:2])
        self.get_sample(cmd_index, preload)

    def get_sample(self, cmd_index, preload):
        if preload:
            default_filename = os.path.join(self.folder_path, 'sample_preload.csv')
            if not os.path.isfile(default_filename):
                print("***No such file!", default_filename)
                # raise IOError("***No such file!", os.path.join(self.folder_path, 'sample_preload.csv'))
                self.get_sample(cmd_index, False)
            else:
                self.X = np.genfromtxt(default_filename, delimiter=',', dtype=np.float32)
            
        else:
            self.X = np.zeros((len(self.file_list_color), self.feature_agent.get_size()))
            i = 0
            for color_file, depth_file in zip(self.file_list_color, self.file_list_depth):
                print(color_file)
                image_color = cv2.imread(color_file, cv2.IMREAD_UNCHANGED)
                image_depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
                self.X[i,:] = self.feature_agent.step(image_color, image_depth)
                i += 1
            
            np.savetxt(os.path.join(self.folder_path, 'sample_preload.csv'), self.X, delimiter=',')
            
        self.read_telemetry(cmd_index)
        print('Load linear regression samples successfully.')

    def read_telemetry(self, index):
        # read telemetry csv
        telemetry_data = np.genfromtxt(os.path.join(self.folder_path, 'airsim.csv'), 
                    delimiter=',', skip_header=True, dtype=np.float32)
        
        self.y = telemetry_data[:, index] # yaw_cmd
        # yaw = np.reshape(telemetry_data[:, index-2], (-1,1)) # yaw
        yawRate = np.reshape(telemetry_data[:, index-1], (-1,1)) # yaw rate
        self.X = np.concatenate((self.X, yawRate), axis=1)

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