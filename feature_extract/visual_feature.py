import numpy as np
import cv2
import os
import glob
import time
# import multiprocessing as mp

class FeatureExtract():
    def __init__(self, config, image_size, printout=False):
        self.config = config
        self.image_size = image_size
        self.printout = printout

        self.configure()

    def configure(self):

        # Sliding Window Init (only executed once)
        image_height = self.image_size[0]
        image_width = self.image_size[1]
        split_size = (self.config['SLIDE_ROWS'], self.config['SLIDE_COLS']) # split size e.g., (#height, #width) (rows, columns)
        
        self.H_points, self.H_size = self.sliding_window(image_height, split_size[0], self.config['SLIDE_OVERLAP'], self.config['SLIDE_FLAG']) 
        self.W_points, self.W_size = self.sliding_window(image_width, split_size[1], self.config['SLIDE_OVERLAP'], self.config['SLIDE_FLAG'])
        
        # Hough Init
        # self.image_gpu = cv2.cuda_GpuMat()
        # self.image_canny = cv2.cuda_GpuMat()
        self.cannyFilter = cv2.cuda.createCannyEdgeDetector(low_thresh=5, high_thresh=20, apperture_size=3)
        self.houghFilter = cv2.cuda.createHoughLinesDetector(rho=1, theta=(np.pi/60), threshold=3, doSort=True, maxLines=32)
        # self.houghResult_gpu = np.zeros(self.config['HOUGH_ANGLES'] * 2 * len(self.H_points) * len(self.W_points))

        # Structure Tensor Init

        # Law Mask Init
        self.create_lawMask(self.config['LAW_MASK'])

        # Optical Flow Init
        self.image_prvs = np.array([], dtype=np.uint8)
        self.nvof = cv2.cuda_FarnebackOpticalFlow.create(numLevels=3, pyrScale=0.5, fastPyramids=False, winSize=15,
                                                    numIters=3, polyN=5, polySigma=1.1, flags=0) 

        # Color Feature Init
        self.feature_list = [
            # Name,              Size,                            Function handle 
            ('hough',            self.config['HOUGH_ANGLES'],     self.hough_feature),
            ('structure_tensor', self.config['TENSOR_HISTBIN'],   self.tensor_feature),
            ('law_mask',         len(self.config['LAW_MASK']),    self.law_feature),
            ('optical_flow',     3,                               self.flow_feature),
        ]

        size_each_window = 0
        self.time_dict = {}
        for name, size, function in self.feature_list:
            size_each_window += size
            self.time_dict[name] = 0.0
        self.feature_color_result = np.zeros(size_each_window * len(self.H_points) * len(self.W_points))
    
        # Depth Feature Init
        self.depth_size = 1
        self.feature_depth_result = np.zeros(self.depth_size * len(self.H_points) * len(self.W_points))
        self.time_dict['depth'] = 0.0

    ''' 
    @ Sliding Window
    '''
    def sliding_window(self, size, split_size, overlap=0.0, flag=0):
        # provide the exact window size
        if flag == 0:
            window_size = min(size, split_size)
        # provide the number of windows
        elif flag == 1:
            if split_size in {0, 1}:
                return [0], size
            window_size = int(size/(split_size-overlap*(split_size-1)))
        else:
            exit('Unknown flag!')

        points = [0]
        stride = int(window_size*(1-overlap))
        counter = 1
        while True:
            pt = stride * counter
            if flag == 0:
                if pt + window_size >= size:
                    points.append(size - window_size)
                    break
                else:
                    points.append(pt)
            else:
                if pt + window_size > size:
                    break
                else:
                    points.append(pt)
            counter += 1

        return points, window_size

    '''
    @ Hough Feature
    '''
    def hough_feature(self, image):
        # Convert to grayscale if necessary
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image_gpu = cv2.cuda_GpuMat(image)
        image_canny = self.cannyFilter.detect(image_gpu)
        houghResult_gpu = self.houghFilter.detect(image_canny)
        houghLines = houghResult_gpu.download()

        hough_result = np.zeros(self.config['HOUGH_ANGLES'])
        if houghLines is not None:
            for i in range(0, len(houghLines[0])):
                thetaIndex = int(houghLines[0][i][1] / (np.pi/15))
                hough_result[thetaIndex] += 1/32

        return hough_result

    '''
    @ Structure Tensor
    '''
    def tensor_feature(self, image):
        # Convert to grayscale if necessary
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
             
        # Apply Sober filter
        Sx = cv2.Sobel(image, cv2.CV_32F, dx=1, dy=0, ksize=3)
        Sy = cv2.Sobel(image, cv2.CV_32F, dx=0, dy=1, ksize=3)
        Sxx = cv2.multiply(Sx, Sx)
        Syy = cv2.multiply(Sy, Sy)
        Sxy = cv2.multiply(Sx, Sy)

        # Apply a box filter
        filter_size = self.config['TENSOR_FILTSIZE']
        # Sxx = cv2.boxFilter(Sxx, cv2.CV_32F, (filter_size, filter_size))
        # Syy = cv2.boxFilter(Syy, cv2.CV_32F, (filter_size, filter_size))
        # Sxy = cv2.boxFilter(Sxy, cv2.CV_32F, (filter_size, filter_size))
        Sxx = cv2.GaussianBlur(Sxx, (filter_size, filter_size), 0)
        Syy = cv2.GaussianBlur(Syy, (filter_size, filter_size), 0)
        Sxy = cv2.GaussianBlur(Sxy, (filter_size, filter_size), 0)

        # Eigenvalue
        tmp1 = Sxx + Syy
        tmp2 = Sxx - Syy
        tmp2 = cv2.multiply(tmp2, tmp2)
        tmp3 = cv2.multiply(Sxy, Sxy)
        tmp4 = np.sqrt(tmp2 + 4.0 * tmp3) 
        lambda1 = 0.5*(tmp1 + tmp4) # biggest eigenvalue
        lambda2 = 0.5*(tmp1 - tmp4) # smallest eigenvalue

        # Coherency
        coherency = cv2.divide(lambda1 - lambda2, lambda1 + lambda2)
        coherency = np.fmax(np.fmin(coherency, 1.0), 0.0)

        # Orientation angle
        orientation_Angle = cv2.phase(Syy-Sxx, 2.0*Sxy, angleInDegrees = True)
        orientation_Angle = 0.5 * orientation_Angle

        # Calculate Histogram
        # orientation_hist = cv2.calcHist(orientation_Angle, channels=[0], mask=None, histSize=[self.config['TENSOR_HISTBIN']], ranges=[0,180],  accumulate = False)
        # orientation_hist = cv2.normalize(orientation_hist, None, alpha=1, beta=None, norm_type=cv2.NORM_L1)
        index = 0
        histbin = self.config['TENSOR_HISTBIN']
        tensor_result = np.zeros(histbin)
        scale = 255 # a scale to normalize the output
        for k in np.linspace(0.0, 180.0, histbin, endpoint=False):
            tmp1 = (orientation_Angle >= k) & (orientation_Angle < k + 180./histbin)
            if tmp1.any():
                tensor_result[index] = np.sum(coherency[tmp1]) / scale
            index += 1

        return tensor_result

    '''
    @ Law Mask
    '''
    def create_lawMask(self, masks):
        # Law's filters
        L3 = np.array([1,2,1], dtype=np.float32)
        E3 = np.array([-1,0,1], dtype=np.float32)
        S3 = np.array([-1,2,-1], dtype=np.float32)
        L5 = np.array([1,4,6,4,1], dtype=np.float32)
        E5 = np.array([-1,-2,0,2,1], dtype=np.float32)
        S5 = np.array([-1,0,2,0,-1], dtype=np.float32)
        W5 = np.array([-1,2,0,-2,1], dtype=np.float32)
        R5 = np.array([1,-4,6,-4,1], dtype=np.float32)

        # A dictionary of law masks
        # The most successful masks are {L5E5, E5S5, R5R5, L5S5, E5L5, S5E5, S5L5}
        lawMask_Dict = {
            "L5L5" : L5.reshape(5,1) * L5,
            "L5E5" : (L5.reshape(5,1) * E5 + E5.reshape(5,1) * L5) / 2,
            "L5S5" : (L5.reshape(5,1) * S5 + S5.reshape(5,1) * L5) / 2,
            "L5R5" : (L5.reshape(5,1) * R5 + R5.reshape(5,1) * L5) / 2,
            "E5E5" : E5.reshape(5,1) * E5,
            "E5S5" : (E5.reshape(5,1) * S5 + S5.reshape(5,1) * E5) / 2,
            "E5R5" : (E5.reshape(5,1) * R5 + R5.reshape(5,1) * E5) / 2,
            "S5S5" : S5.reshape(5,1) * S5,
            "S5R5" : (S5.reshape(5,1) * R5 + R5.reshape(5,1) * S5) / 2,
            "R5R5" : R5.reshape(5,1) * R5,
        }

        self.law_masks = []
        for name in masks:
            self.law_masks.append((name, lawMask_Dict[name]))

    def law_feature(self, image):
        if len(image.shape) == 3:
            # Convert to YCrCb colorspace
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

            # Apply Law's masks
            law_result = np.zeros(len(self.law_masks))
            index = 0
            scale = 255 # a scale to normalize the output
            for name, mask in self.law_masks:
                if name == "L5L5":
                    for j in range(0,3):
                        image_filtered = cv2.filter2D(image[:,:,j], cv2.CV_32F, mask)
                        # image_filtered = cv2.normalize(image_filtered, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        law_result[index] = np.mean(abs(image_filtered)) / scale
                        index += 1            
                else:
                    image_filtered = cv2.filter2D(image[:,:,0], cv2.CV_32F, mask)
                    # image_filtered = cv2.normalize(image_filtered, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    law_result[index] = np.mean(abs(image_filtered)) / scale
                    index += 1    
        else:
            # if image is in grayscale
            law_result = np.zeros(len(self.law_masks))
            index = 0
            for name, mask in self.law_masks:
                image_filtered = cv2.filter2D(image[:,:,0], cv2.CV_32F, mask)
                # image_filtered = cv2.normalize(image_filtered, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                law_result[index] = np.mean(abs(image_filtered)) / scale
                index += 1    

        return law_result

    '''
    @ Optical Flow
    '''
    def flow_feature(self, image_next, image_prvs):
        # Convert to grayscale
        image_next = cv2.cvtColor(image_next, cv2.COLOR_BGR2GRAY)
        image_prvs = cv2.cvtColor(image_prvs, cv2.COLOR_BGR2GRAY)

        image_next = cv2.cuda_GpuMat(image_next)
        image_prvs = cv2.cuda_GpuMat(image_prvs)

        # Call OF function
        flow_gpu = self.nvof.calc(image_prvs, image_next, None)
        flow = flow_gpu.download()

        # Calculate magnitude and angle
        # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag = np.sqrt(np.square(flow[...,0]) + np.square(flow[...,1])) 
        
        flow_result = np.zeros(3)
        flow_result[0] = np.amax(mag)
        flow_result[1] = np.amin(mag)
        flow_result[2] = np.mean(mag)
        
        return flow_result

    '''
    @ Depth feature
    '''
    def depth_feature(self, image, reverse=True):
        # Reverse the pixel value
        if reverse:
            image = 255 - image
        
        # Currently just the average over the entire window
        depth_result = np.mean(np.abs(image)) / 255. # in range [0.0, 1.0]

        return depth_result

    '''
    @ Update function
    '''
    def update_color(self, image):
        # Convert to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Get features for each window
        k = 0
        for i in self.H_points:
            for j in self.W_points:
                # Cropped image
                cropped_image = image[i:i+self.H_size, j:j+self.W_size]
                for name, size, function in self.feature_list:
                    start_time = time.perf_counter()
                    if name is 'optical_flow':
                        cropped_image_prvs = self.image_prvs[i:i+self.H_size, j:j+self.W_size]
                        self.feature_color_result[k:k+size] = function(cropped_image, cropped_image_prvs)
                    else:
                        self.feature_color_result[k:k+size] = function(cropped_image)
                    self.time_dict[name] = time.perf_counter() - start_time 
                    k += size
        # for optical flow
        self.image_prvs = image
        return self.feature_color_result

    # def update_color_multicore(self, image):
    #     # Convert to BGR
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
    #     # Get features for each window
    #     k = 0
    #     for i in self.H_points:
    #         for j in self.W_points:
    #             # Cropped image
    #             cropped_image = image[i:i+self.H_size, j:j+self.W_size]

    #             tmp = mp.Queue()
    #             jobs = []
    #             for name, size, function in self.feature_list:
    #                 start_time = time.perf_counter()
    #                 # if name is 'optical_flow':
    #                 #     cropped_image_prvs = self.image_prvs[i:i+self.H_size, j:j+self.W_size]
    #                 #     self.feature_color_result[k:k+size] = function(cropped_image, cropped_image_prvs)
    #                 # else:
    #                 #     self.feature_color_result[k:k+size] = function(cropped_image)
    #                 # self.time_dict[name] = time.perf_counter() - start_time 
    #                 # k += size

    #                 if name is 'optical_flow':
    #                     cropped_image_prvs = self.image_prvs[i:i+self.H_size, j:j+self.W_size]
    #                     task = mp.Process(target=function, args=(cropped_image,cropped_image_prvs))
    #                 else:
    #                     task = mp.Process(target=function, args=(cropped_image,))
    #                 jobs.append(task)
    #                 task.start()
    #                 self.time_dict[name] = time.perf_counter() - start_time 

    #             for proc in jobs:
    #                 proc.join()


        # for optical flow
        self.image_prvs = image
        return self.feature_color_result

    def update_depth(self, image):
        # Get features for each window
        k = 0
        for i in self.H_points:
            for j in self.W_points:    
                # Cropped images          
                cropped_image = image[i:i+self.H_size, j:j+self.W_size]
                start_time = time.perf_counter()
                self.feature_depth_result[k:k+self.depth_size] = self.depth_feature(cropped_image)
                self.time_dict['depth'] = time.perf_counter() - start_time 
                k += self.depth_size

        return self.feature_depth_result

    def step(self, image_color, image_depth):
        # for the optical flow
        if self.image_prvs.size == 0:
            self.image_prvs = image_color
    
        self.update_color(image_color)
        self.update_depth(image_depth)
        # Update time for each feature functions
        self.format_time(self.printout)
        return np.concatenate((self.feature_color_result, self.feature_depth_result))

    def format_time(self, printout=True):
        # Format the elapsed time and print out 
        factor = 1.0 / sum(self.time_dict.values())
        for key in self.time_dict:
            self.time_dict[key] *= factor
            if printout:
                print('{:s}: {:.2%},'.format(key, self.time_dict[key]), end=" ")        
        if printout:
            print('')

    def get_size(self):
        return len(self.feature_color_result) + len(self.feature_depth_result)

    def reset(self):
        self.image_prvs = np.array([], dtype=np.uint8)
        self.feature_color_result.fill(0.0)
        self.feature_depth_result.fill(0.0)