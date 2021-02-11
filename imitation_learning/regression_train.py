import glob
import os
import numpy as np
import cv2
import pandas
import multiprocessing as mp

from feature_extract import *
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from scipy.optimize import curve_fit

# Exponential decaying function
def exponential_decay(num_prvs=5, max_prvs=15, ratio=1.5):
    y = []
    for t in range(0, num_prvs):
        y.append(int(np.ceil(max_prvs * np.exp(-t/ratio))))
    return y

def calculate_regression(X, y):
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


class RegTrain_single():
    '''
    Linear Regression Training Agent in single core
    '''
    def __init__(self, folder_path, output_path, num_prvs, image_size, preload=False, printout=False):
        self.image_size = image_size
        self.num_prvs = num_prvs
        tmp_feature_agent = FeatureExtract(feature_config, image_size)
        self.X = np.empty((0, tmp_feature_agent.get_size()+self.num_prvs+1))
        self.y = np.empty((0,))

        self.run(folder_path, output_path, preload)

    def run(self, folder_path, output_path, preload):
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
            if y is not None:
                # ensure that there is no inf term
                if np.sum(X==np.inf) > 0:
                    print('*** Got Inf in the feature vector!')
                    X[X==np.inf] = 0
            
                self.X = np.concatenate((self.X, X), axis=0)
                self.y = np.concatenate((self.y, y), axis=0)

        # Train the model
        self.train()
        
        # Save the weight to file
        self.save_weight(os.path.join(output_path, 'reg_weight.csv'))


    def get_sample(self, file_list_color, file_list_depth, folder_path, preload):
        # read from telemetry file
        X_extra, y = self.read_telemetry(folder_path)
        
        if y is not None:
            file_path = os.path.join(folder_path, 'feature_preload.pkl')
            feature_agent = FeatureExtract(feature_config, self.image_size)

            if preload and os.path.isfile(file_path):
                X = pandas.read_pickle(file_path).to_numpy()
            else:
                X = np.zeros((len(file_list_color), feature_agent.get_size()))
                i = 0
                for color_file, depth_file in zip(file_list_color, file_list_depth):
                    # print(color_file)
                    image_color = cv2.imread(color_file, cv2.IMREAD_UNCHANGED)
                    image_depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
                    X[i,:] = feature_agent.step(image_color, image_depth)
                    i += 1

                # save to file for the future use
                pandas.DataFrame(X).to_pickle(file_path) 
        
            X = np.column_stack((X, X_extra))
            print('Load samples from {:s} successfully.'.format(folder_path))  
            return X, y
        else:
            return None, None

    def read_telemetry(self, folder_path):
        # read telemetry csv       
        telemetry_data = pandas.read_csv(os.path.join(folder_path, 'airsim.csv'))

        if telemetry_data.iloc[-1,0] == 'crashed':
            print('Find crashed dataset in {:s}'.format(folder_path))
            return None, None

        # Yaw cmd
        y = telemetry_data['yaw_cmd'][:-1].to_numpy()

        # Previous commands with time decaying
        y_prvs = np.zeros((len(y), self.num_prvs))
        # prvs_index = exp_decay(self.num_prvs)
        prvs_index = [5,4,3,2,1]
        
        for i in range(len(y)):
            for j in range(self.num_prvs):
                y_prvs[i,j] = y[max(i-prvs_index[j], 0)]

        # Yaw rate
        yawRate = np.reshape(telemetry_data['yaw_rate'][:-1].to_numpy(), (-1,1))
        X_extra = np.concatenate((y_prvs, yawRate), axis=1)
        return X_extra, y

    def calculate_weight(self):
        weight, intercept, self.r2, self.rmse = calculate_regression(self.X, self.y)
        self.weight = np.append(intercept, weight)
        
    def train(self):
        self.calculate_weight()
        print('Trained linear regression model successfully.')

    def save_weight(self, file_path):
        np.savetxt(file_path, self.weight, delimiter=',')
        print('Save weight to: ', file_path)


class RegTrain_multi(RegTrain_single):
    '''
    Linear Regression Training Agent in multiple cores
    '''
    def __init__(self, folder_path, output_path, num_prvs, image_size, preload=False, printout=False):
        super().__init__(folder_path, output_path, num_prvs, image_size, preload, printout)

    # Override function
    def run(self, folder_path, output_path, preload):
        jobs = []
        pool = mp.Pool(min(12, mp.cpu_count())) # use how many cores
        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            file_list_color = glob.glob(os.path.join(subfolder_path, 'color', '*.png'))
            file_list_depth = glob.glob(os.path.join(subfolder_path, 'depth', '*.png'))
            file_list_color.sort()
            file_list_depth.sort()
            if len(file_list_color) != len(file_list_depth):
                raise Exception("The size of color and depth images does not match!")
            
            jobs.append(pool.apply_async(self.get_sample, args=(file_list_color, file_list_depth, subfolder_path, preload)))

        # Wait results
        results = [proc.get() for proc in jobs] 

        # Visual feature
        for X, y in results:
            if y is not None:
                # ensure that there is no inf term
                if np.sum(X==np.inf) > 0:
                    print('*** Got Inf in the feature vector!')
                    X[X==np.inf] = 0
            
                self.X = np.concatenate((self.X, X), axis=0)
                self.y = np.concatenate((self.y, y), axis=0)

        # Train the model
        self.train()
        
        # Save the weight to file
        self.save_weight(os.path.join(output_path, 'reg_weight.csv'))