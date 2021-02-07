import glob
import os
import numpy as np
import cv2
import pandas

from feature_extract import *
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from scipy.optimize import curve_fit

def calculate_regression(X, y, disp_summary=False):
    print('\n*** Training Results ***')
    reg = LinearRegression().fit(X, y)
    r_square = reg.score(X, y)
    print('r_square = {:.6f}'.format(r_square))

    mse = np.mean((reg.predict(X)-y)**2)
    rmse = np.sqrt(mse)
    print('MSE = {:.6f}'.format(mse))
    print('RMSE = {:.6f}'.format(rmse))
    print('Number of weights = {:} '.format(len(reg.coef_)+1))
    print('***********************\n')
    return reg.coef_, reg.intercept_, r_square, rmse

def sigmoid_function(X, w):
    print(w.shape)
    print(X.shape)
    tmp = np.dot(w[:2070], X) + w[2071]
    print(tmp.shape)
    y_pred = w[2072] / (1.0 + np.exp(-w[2073]*tmp)) + w[2074]
    return y_pred

def calculate_nonlinear(X, y):
    a = 0
    b = 1
    c = 1
    d = 1
    popt, pcov  = curve_fit(sigmoid_function, xdata=X, ydata=y, p0=np.ones((2074,)))

class RegTrain():
    '''
    Linear Regression Training Agent
    '''
    def __init__(self, folder_path, image_size, cmd_index, preload=False, printout=False):
        self.feature_agent = FeatureExtract(feature_config, image_size, printout)
        self.cmd_index = cmd_index
        self.cmd_numprvs = feature_config['CMD_NUMPRVS']
        self.cmd_decay = feature_config['CMD_DECAY']
        self.X = np.empty((0, self.feature_agent.get_size()+self.cmd_numprvs+1))
        self.y = np.empty((0,))

        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            file_list_color = glob.glob(os.path.join(subfolder_path, 'color', '*.png'))
            file_list_depth = glob.glob(os.path.join(subfolder_path, 'depth', '*.png'))
            file_list_color.sort()
            file_list_depth.sort()
            if len(file_list_color) != len(file_list_depth):
                raise Exception("The size of color and depth images does not match!")
            
            # Visual feature
            X, y = self.get_sample(file_list_color, file_list_depth, subfolder_path, preload)
            # ensure that there is no inf term
            if np.sum(X==np.inf) > 0:
                print('*** Got Inf in the feature vector!')
                X[X==np.inf] = 0
            
            self.X = np.concatenate((self.X, X), axis=0)
            self.y = np.concatenate((self.y, y), axis=0)

    def get_sample(self, file_list_color, file_list_depth, folder_path, preload):
        file_path = os.path.join(folder_path, 'feature_preload.pkl')
        
        if preload and os.path.isfile(file_path):
            X = pandas.read_pickle(file_path).to_numpy()
        else:
            X = np.zeros((len(file_list_color), self.feature_agent.get_size()))
            i = 0
            for color_file, depth_file in zip(file_list_color, file_list_depth):
                image_color = cv2.imread(color_file, cv2.IMREAD_UNCHANGED)
                image_depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
                X[i,:] = self.feature_agent.step(image_color, image_depth)
                self.feature_agent.reset()
                i += 1

            # save to file for the future use
            pandas.DataFrame(X).to_pickle(file_path)

        print('Load samples from {:s} successfully.'.format(folder_path))     

        X_extra, y = self.read_telemetry(folder_path, self.cmd_index)
        X = np.column_stack((X, X_extra))
        return X, y

    def read_telemetry(self, folder_path, index):
        # read telemetry csv
        telemetry_data = np.genfromtxt(os.path.join(folder_path, 'airsim.csv'), 
                    delimiter=',', skip_header=True)
        
        telemetry_data1 = pandas.read_csv(os.path.join(folder_path, 'airsim.csv'))
        y = telemetry_data1['yaw_cmd'].to_numpy()
        for k in range(len(y)):
            if np.abs(y[k]) < 1e-3:
                y[k] = 0.0

        # Previous commands with time decaying
        y_prvs = np.zeros((len(y), self.cmd_numprvs))
        for i in range(len(y)):
            for j in range(self.cmd_numprvs):
                if i > j:
                    y_prvs[i,j] = y[i-(j+1)] * self.cmd_decay**(j+1)

        # Yaw rate
        print(telemetry_data1['yaw_rate'].to_numpy().shape)
        yawRate =  np.reshape(telemetry_data1['yaw_rate'].to_numpy(), (-1,1))  # yaw rate
        X_extra = np.concatenate((y_prvs, yawRate), axis=1)
        return X_extra, y

    def calculate_weight(self):
        weight, intercept, self.r2, self.rmse = calculate_regression(self.X, self.y)
        self.weight = np.append(intercept, weight)
        
    def train(self):
        self.calculate_weight()

    def save_weight(self, file_path):
        np.savetxt(file_path, self.weight, delimiter=',')
        print('Save weight to: ', file_path)