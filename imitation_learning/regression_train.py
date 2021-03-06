import glob
import os
import numpy as np
import cv2
import pandas
import multiprocessing as mp
from tqdm import tqdm 
import pickle
import math

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
    if method == 'LinearRegression':
        model = LinearRegression(normalize=True).fit(X, y)
    elif method == 'Ridge':
        model = Ridge(normalize=True).fit(X, y)
    elif method == 'BayesianRidge':
        model = BayesianRidge(normalize=True).fit(X, y)
    elif method == 'Sigmoid':
        pass
    # elif method == 'Polynomial':
    # print(PolynomialFeatures(degree=3).fit(X, y))
    else:
        exit('Unknown regression method: {:s}'.format(method))

    print('Regression type = {:s}'.format(str(model)))
    r_square = model.score(X, y)
    print('R_square = {:.6f}'.format(r_square))
    mse = np.mean((model.predict(X)-y)**2)
    rmse = np.sqrt(mse)
    # print('MSE = {:.6f}'.format(mse))
    print('RMSE = {:.6f}'.format(rmse))
    print('Number of weights = {:} '.format(len(model.coef_)+1))
    print('Number of points = {:} '.format(len(y)))
    print('***********************\n')

    result = {'Model': model, 'R_square':r_square, 'RMSE':rmse}
    return result

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
    def __init__(self, **kwargs):
        self.dataset_dir = kwargs['dataset_dir']
        self.output_dir = kwargs['output_dir']
        self.weight_filename = kwargs['weight_filename']
        self.model_filename = kwargs['model_filename']
        self.num_prvs = kwargs['num_prvs']
        self.prvs_mode = kwargs['prvs_mode']
        self.image_size = kwargs['image_size']
        self.reg_type = kwargs['reg_type']
        self.preload = kwargs.get('preload', True)
        self.printout = kwargs.get('printout', False)

        self.X = np.empty((0, FeatureExtract.get_size(feature_config, self.image_size) + self.num_prvs + 1))
        self.y = np.empty((0,))

        # Main function
        self.run()

    def run(self):
        print('Loading datasets...')
        for subfolder in tqdm(os.listdir(self.dataset_dir)):
            subfolder_dir = os.path.join(self.dataset_dir, subfolder)
            file_list_color = glob.glob(os.path.join(subfolder_dir, 'color', '*.png'))
            file_list_depth = glob.glob(os.path.join(subfolder_dir, 'depth', '*.png'))
            file_list_color.sort()
            file_list_depth.sort()
            if len(file_list_color) != len(file_list_depth):
                raise Exception("The size of color and depth images does not match!")
            
            # Get feature vector
            X, y = self.get_sample(file_list_color, file_list_depth, subfolder_dir)
            if y is not None:
                # ensure that there is no inf term
                if np.sum(X==np.inf) > 0:
                    print('*** Got Inf in the feature vector in {:s}!'.format(subfolder))
                    X[X==np.inf] = 0.0
            
                self.X = np.concatenate((self.X, X), axis=0)
                self.y = np.concatenate((self.y, y), axis=0)

        print('Load datasets successfully.')
        
        # Train the model
        model, weight = self.train()
        
        # Save result to file
        self.save_result(model, weight)

    def get_sample(self, file_list_color, file_list_depth, subfolder_dir):
        # Read from telemetry file
        X_extra, y, N = self.read_telemetry(subfolder_dir, self.num_prvs)

        if N is not None:
            file_path = os.path.join(subfolder_dir, 'feature_preload.pkl') # default file name
            feature_agent = FeatureExtract(feature_config, self.image_size, self.printout)

            if self.preload and os.path.isfile(file_path):
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
            # print('Load samples from {:s} successfully.'.format(subfolder_dir))  
            return X, y
        else:
            return None, None

    def read_telemetry(self, folder_dir, num_prvs):
        # Read telemetry csv       
        telemetry_data = pandas.read_csv(os.path.join(folder_dir, 'airsim.csv'))
        N = len(telemetry_data) - 1 # length of data 

        if telemetry_data.iloc[-1,0] == 'crashed':
            print('Find crashed dataset in {:s}'.format(folder_dir))
            N -= (5 * 10) # remove the last 5 seconds data
            if N < 0:
                return None, None, None

        # Yaw cmd
        y = telemetry_data['yaw_cmd'][:N].to_numpy()

        # Yaw rate
        yawRate = np.reshape(telemetry_data['yaw_rate'][:N].to_numpy(), (-1,1))
        yawRate_norm = yawRate * (180.0 / math.pi) / 45.0

        # Previous commands with time decaying
        if num_prvs > 0:
            y_prvs = np.zeros((N, num_prvs))
            if self.prvs_mode == 'exponential':
                prvs_index = exponential_decay(num_prvs)
            elif self.prvs_mode == 'linear':
                prvs_index = [i for i in reversed(range(1, num_prvs+1))]
            else:
                raise Exception('Unknown prvs_mode {:s}'.format(self.prvs_mode))
            for i in range(N):
                for j in range(num_prvs):
                    y_prvs[i,j] = y[max(i-prvs_index[j], 0)]

            # X_extra = [y_prvs, yawRate]
            X_extra = np.concatenate((y_prvs, yawRate_norm), axis=1)
        else:
            X_extra = yawRate_norm
        
        return X_extra, y, N
       
    def train(self):
        result = calculate_regression(self.X, self.y, method=self.reg_type)

        if self.reg_type in ['Ridge', 'LinearRegression']:
            weight = np.append(result['Model'].intercept_, result['Model'].coef_)
        else:
            weight = None
        
        print('Trained linear regression model successfully.')
        return result, weight

    def save_result(self, result, weight):
        pickle.dump(result, open(os.path.join(self.output_dir, self.model_filename), 'wb'))
        print('Save model to: ', os.path.join(self.output_dir, self.model_filename))

        if weight is not None:
            np.savetxt(os.path.join(self.output_dir, self.weight_filename), weight, delimiter=',')
            print('Save weight to: ', os.path.join(self.output_dir, self.weight_filename))

class RegTrain_multi(RegTrain_single):
    """
    Linear Regression Training Agent with multiple cores
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # Override function
    def run(self):
        jobs = []
        pool = mp.Pool(min(12, mp.cpu_count())) # use how many cores
        for subfolder in os.listdir(self.dataset_dir):
            subfolder_dir = os.path.join(self.dataset_dir, subfolder)
            file_list_color = glob.glob(os.path.join(subfolder_dir, 'color', '*.png'))
            file_list_depth = glob.glob(os.path.join(subfolder_dir, 'depth', '*.png'))
            file_list_color.sort()
            file_list_depth.sort()
            if len(file_list_color) != len(file_list_depth):
                raise Exception("The size of color and depth images does not match!")
            
            jobs.append(pool.apply_async(self.get_sample, args=(file_list_color, file_list_depth, subfolder_dir)))

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
        model, weight = self.train()
        
        # Save result to file
        self.save_result(model, weight)

class RegTrain_single_advanced():
    """
    Linear Regression Training Agent with Single Core
    """
    def __init__(self, **kwargs):
        self.dataset_dir = kwargs['dataset_dir']
        self.output_dir = kwargs['output_dir']
        self.weight_filename = kwargs['weight_filename']
        self.model_filename = kwargs['model_filename']
        self.num_prvs = kwargs['num_prvs']
        self.prvs_mode = kwargs['prvs_mode']
        self.image_size = kwargs['image_size']
        self.reg_type = kwargs['reg_type']
        self.preload = kwargs.get('preload', True)
        self.printout = kwargs.get('printout', False)
        self.subject = kwargs.get('subject')
        self.map_list = kwargs.get('map_list')
        self.iteration = kwargs.get('iteration')

        if self.num_prvs == 0:
            self.X = np.empty((0, FeatureExtract.get_size(feature_config, self.image_size)))
        else:
            self.X = np.empty((0, FeatureExtract.get_size(feature_config, self.image_size) + self.num_prvs + 1))
        self.y = np.empty((0,))

        # Main function
        self.run()

    def run(self):
        print('Loading datasets...')
        iteration = 'iter' + str(self.iteration)
        for map in os.listdir(os.path.join(self.dataset_dir, self.subject)):
            if map in self.map_list:
                folder_path = os.path.join(self.dataset_dir, self.subject, map, iteration)
                for subfolder in os.listdir(folder_path):
                    subfolder_dir = os.path.join(folder_path, subfolder)
                    file_list_color = glob.glob(os.path.join(subfolder_dir, 'color', '*.png'))
                    file_list_depth = glob.glob(os.path.join(subfolder_dir, 'depth', '*.png'))
                    file_list_color.sort()
                    file_list_depth.sort()
                    if len(file_list_color) != len(file_list_depth):
                        raise Exception("The size of color and depth images does not match!")
                    
                    # Get feature vector
                    X, y = self.get_sample(file_list_color, file_list_depth, subfolder_dir, iteration)
                    if y is not None:
                        # ensure that there is no inf term
                        if np.sum(X==np.inf) > 0:
                            print('*** Got Inf in the feature vector in {:s}!'.format(subfolder))
                            X[X==np.inf] = 0.0
                    
                        self.X = np.concatenate((self.X, X), axis=0)
                        self.y = np.concatenate((self.y, y), axis=0)

        print('Load datasets successfully.')
        
        # Train the model
        model, weight = self.train()
        
        # Save result to file
        self.save_result(model, weight)

    def get_sample(self, file_list_color, file_list_depth, subfolder_dir, iteration):
        # Read from telemetry file
        X_extra, y, N, pilot_index = self.read_telemetry(subfolder_dir, self.num_prvs, iteration)

        if N is not None:
            file_path = os.path.join(subfolder_dir, 'feature_preload.pkl') # default file name
            feature_agent = FeatureExtract(feature_config, self.image_size, self.printout)

            if self.preload and os.path.isfile(file_path):
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
            if X_extra is not None:
                X = np.column_stack((X[pilot_index,:], X_extra[pilot_index,:]))
            else:
                X = X[pilot_index,:]
            # print('Load samples from {:s} successfully.'.format(subfolder_dir))  
            # print(X.shape, y.shape)
            return X, y[pilot_index]
        else:
            return None, None

    def read_telemetry(self, folder_dir, num_prvs, iteration=0):
        # Read telemetry csv       
        telemetry_data = pandas.read_csv(os.path.join(folder_dir, 'airsim.csv'))
        N = len(telemetry_data) - 1 # length of data 

        if telemetry_data.iloc[-1,0] == 'crashed':
            print('Find crashed dataset in {:s}'.format(folder_dir))
            N -= (5 * 10) # remove the last 5 seconds data
            if N < 0:
                return None, None, None, None

        # Yaw cmd
        y = telemetry_data['yaw_cmd'][:N].to_numpy()

        # Yaw rate
        yawRate = np.reshape(telemetry_data['yaw_rate'][:N].to_numpy(), (-1,1))
        yawRate_norm = yawRate * (180.0 / math.pi) / 45.0

        # Previous commands with time decaying
        if num_prvs > 0:
            y_prvs = np.zeros((N, num_prvs))
            if self.prvs_mode == 'exponential':
                prvs_index = exponential_decay(num_prvs)
            elif self.prvs_mode == 'linear':
                prvs_index = [i for i in reversed(range(1, num_prvs+1))]
            else:
                raise Exception('Unknown prvs_mode {:s}'.format(self.prvs_mode))
            for i in range(N):
                for j in range(num_prvs):
                    y_prvs[i,j] = y[max(i-prvs_index[j], 0)]

            # X_extra = [y_prvs, yawRate]
            X_extra = np.concatenate((y_prvs, yawRate_norm), axis=1)
        else:
            X_extra = None
        
        # flag
        flag = telemetry_data['flag'][:N].to_numpy()
        pilot_index = (flag == 0)
        return X_extra, y, N, pilot_index
       
    def train(self):
        result = calculate_regression(self.X, self.y, method=self.reg_type)

        if self.reg_type in ['Ridge', 'LinearRegression']:
            weight = np.append(result['Model'].intercept_, result['Model'].coef_)
        else:
            weight = None
        
        print('Trained linear regression model successfully.')
        return result, weight

    def save_result(self, result, weight):
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        pickle.dump(result, open(os.path.join(self.output_dir, self.model_filename), 'wb'))
        print('Save model to: ', os.path.join(self.output_dir, self.model_filename))

        if weight is not None:
            np.savetxt(os.path.join(self.output_dir, self.weight_filename), weight, delimiter=',')
            print('Save weight to: ', os.path.join(self.output_dir, self.weight_filename))

class RegTrain_multi_advanced(RegTrain_single_advanced):
    """
    Linear Regression Training Agent with Multi Core
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self):
        print('Loading datasets...')
        jobs = []    
        pool = mp.Pool(min(12, mp.cpu_count())) # use how many cores
        for iteration in range(self.iteration+1):
            for map in os.listdir(os.path.join(self.dataset_dir, self.subject)):
                if map in self.map_list:
                    folder_path = os.path.join(self.dataset_dir, self.subject, map, 'iter' + str(iteration))
                    if os.path.isdir(folder_path):
                        for subfolder in os.listdir(folder_path):
                            subfolder_dir = os.path.join(folder_path, subfolder)
                            file_list_color = glob.glob(os.path.join(subfolder_dir, 'color', '*.png'))
                            file_list_depth = glob.glob(os.path.join(subfolder_dir, 'depth', '*.png'))
                            file_list_color.sort()
                            file_list_depth.sort()
                            if len(file_list_color) != len(file_list_depth):
                                raise Exception("The size of color and depth images does not match!")
                            
                            jobs.append(pool.apply_async(self.get_sample, args=(file_list_color, file_list_depth, subfolder_dir, iteration)))
                
        # Wait results
        results = [proc.get() for proc in tqdm(jobs)] 

        for X, y in results:
            if y is not None:
                # ensure that there is no inf term
                if np.sum(X==np.inf) > 0:
                    print('*** Got Inf in the feature vector in {:s}!'.format(subfolder))
                    X[X==np.inf] = 0.0
            
                self.X = np.concatenate((self.X, X), axis=0)
                self.y = np.concatenate((self.y, y), axis=0)

        print('Load datasets successfully.')
        
        # Train the model
        model, weight = self.train()
        
        # Save result to file
        self.save_result(model, weight)
