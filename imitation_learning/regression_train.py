import glob
import os
import numpy as np
import cv2
import pandas
import multiprocessing as mp
from tqdm import tqdm 
import pickle

from feature_extract import *
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit

# Exponential decaying function
def exponential_decay(num_prvs=5, max_prvs=15, ratio=1.5):
    y = []
    for t in range(0, num_prvs):
        y.append(int(np.ceil(max_prvs * np.exp(-t/ratio))))
    return y

def calculate_regression(X, y, method='Ridge'):
    print('\n*** Training Results ***')
    if method is 'LinearRegression':
        reg = LinearRegression(normalize=True).fit(X, y)
    if method is 'Ridge':
        reg = Ridge(normalize=True).fit(X, y)
    if method is 'BayesianRidge':
        reg = BayesianRidge(normalize=True).fit(X, y)
    # if method is 'Polynomial':
    # print(PolynomialFeatures(degree=3).fit(X, y))
    if method is 'Sigmoid':
        pass
    r_square = reg.score(X, y)
    print('r_square = {:.6f}'.format(r_square))

    mse = np.mean((reg.predict(X)-y)**2)
    rmse = np.sqrt(mse)
    print('MSE = {:.6f}'.format(mse))
    print('RMSE = {:.6f}'.format(rmse))
    print('Number of weights = {:} '.format(len(reg.coef_)+1))
    print('***********************\n')
    return reg, r_square, rmse

# def sigmoid(x, Beta_1, Beta_2): 
#      y = 1. / (1. + np.exp(-Beta_1*(x-Beta_2))) 
#      return y 

def sigmoid_nonlinear(X, W, beta1, beta2):
    tmp = 0
    for i in range(0, 2071):
        tmp += X[i] * W[i]
    y = 2. / (1. + np.exp(-beta1*(tmp-beta2))) - 1.0
    return y

def calculate_nonlinear(X, y):
    popt, pcov  = curve_fit(sigmoid_nonlinear, xdata=X, ydata=y, p0=np.ones((2072,)))

class RegTrain_single():
    """
    Linear Regression Training Agent with Single Core
    """
    def __init__(self, folder_path, output_path, weight_filename, num_prvs, image_size, preload=False, printout=False):
        self.X = np.empty((0, FeatureExtract.get_size(feature_config, image_size) + num_prvs + 1))
        self.y = np.empty((0,))

        # Main function
        self.run(folder_path, output_path, weight_filename, num_prvs, image_size, preload, printout)

    def run(self, folder_path, output_path, weight_filename, num_prvs, image_size, preload, printout):
        for subfolder in tqdm(os.listdir(folder_path)):
            subfolder_path = os.path.join(folder_path, subfolder)
            file_list_color = glob.glob(os.path.join(subfolder_path, 'color', '*.png'))
            file_list_depth = glob.glob(os.path.join(subfolder_path, 'depth', '*.png'))
            file_list_color.sort()
            file_list_depth.sort()
            if len(file_list_color) != len(file_list_depth):
                raise Exception("The size of color and depth images does not match!")
            
            # Get feature vector
            X, y = self.get_sample(file_list_color, file_list_depth, subfolder_path, num_prvs, image_size, preload, printout)
            if y is not None:
                # ensure that there is no inf term
                if np.sum(X==np.inf) > 0:
                    print('*** Got Inf in the feature vector in {:s}!'.format(subfolder))
                    X[X==np.inf] = 0.0
            
                self.X = np.concatenate((self.X, X), axis=0)
                self.y = np.concatenate((self.y, y), axis=0)

        # Train the model
        self.train()
        
        # Save weight to file
        self.save_weight(output_path, weight_filename)

    def get_sample(self, file_list_color, file_list_depth, folder_path, num_prvs, image_size, preload, printout):
        # Read from telemetry file
        X_extra, y, N = self.read_telemetry(folder_path, num_prvs)

        if N is not None:
            file_path = os.path.join(folder_path, 'feature_preload.pkl') # default file name
            feature_agent = FeatureExtract(feature_config, image_size, printout)

            if preload and os.path.isfile(file_path):
                X = pandas.read_pickle(file_path).to_numpy()
            else:
                X = np.zeros((N, len(feature_agent.feature_result)))
                i = 0
                for color_file, depth_file in zip(file_list_color, file_list_depth):
                    image_color = cv2.imread(color_file, cv2.IMREAD_UNCHANGED)
                    image_depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
                    # tic = time.perf_counter()
                    X[i,:] = feature_agent.step(image_color, image_depth, 'BGR')
                    # print('Elapsed time = {:.5f} sec'.format(time.perf_counter()-tic))
                    i += 1
                    if i >= N:
                        break

                # Save to file for future use
                pandas.DataFrame(X).to_pickle(file_path) 

            # Combine output
            X = np.column_stack((X, X_extra))
            # print('Load samples from {:s} successfully.'.format(folder_path))  
            return X, y
        else:
            return None, None

    def read_telemetry(self, folder_path, num_prvs):
        # Read telemetry csv       
        telemetry_data = pandas.read_csv(os.path.join(folder_path, 'airsim.csv'))
        N = len(telemetry_data) - 1 # length of data 

        if telemetry_data.iloc[-1,0] == 'crashed':
            print('Find crashed dataset in {:s}'.format(folder_path))
            N -= (5 * 10) # remove the last 3 sec data
            if N < 0:
                return None, None, None

        # Yaw cmd
        y = telemetry_data['yaw_cmd'][:N].to_numpy()

        # Previous commands with time decaying
        y_prvs = np.zeros((N, num_prvs))
        # prvs_index = exponential_decay(num_prvs)
        prvs_index = [i for i in reversed(range(1, num_prvs+1))]
        for i in range(N):
            for j in range(num_prvs):
                y_prvs[i,j] = y[max(i-prvs_index[j], 0)]

        # Yaw rate
        yawRate = np.reshape(telemetry_data['yaw_rate'][:N].to_numpy(), (-1,1))
        
        # X_extra = [y_prvs, yawRate]
        X_extra = np.concatenate((y_prvs, yawRate), axis=1)
        
        return X_extra, y, N
       
    def train(self):
        self.model, r2, rmse = calculate_regression(self.X, self.y, method='Ridge')
        self.weight = np.append(self.model.intercept_, self.model.coef_)

        # calculate_nonlinear(self.X, self.y)
        print('Trained linear regression model successfully.')

    def save_weight(self, output_path, filename):
        pickle.dump(self.model, open(os.path.join(output_path, 'reg_model.pkl'), 'wb'))
        print('Save model to: ', os.path.join(output_path, 'reg_model.pkl'))

        np.savetxt(os.path.join(output_path, filename), self.weight, delimiter=',')
        print('Save weight to: ', os.path.join(output_path, filename))

class RegTrain_multi(RegTrain_single):
    """
    Linear Regression Training Agent with multiple cores
    """
    def __init__(self, folder_path, output_path, weight_filename, num_prvs, image_size, preload=False, printout=False):
        super().__init__(folder_path, output_path, weight_filename, num_prvs, image_size, preload, printout)

    # Override function
    def run(self, folder_path, output_path, weight_filename, num_prvs, image_size, preload, printout):
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
            
            jobs.append(pool.apply_async(self.get_sample, args=(file_list_color, file_list_depth, subfolder_path, \
                                                             num_prvs, image_size, preload, printout)))

        # Wait results
        results = [proc.get() for proc in tqdm(jobs)] 

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
        self.save_weight(output_path, weight_filename)