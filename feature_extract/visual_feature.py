import numpy as np
import cv2
import os
import glob

class FeatureExtract():
    def __init__(self, config, image_size):
        self.config = config

        # Sliding Window Init (only executed once)
        image_height = image_size[0]
        image_width = image_size[1]
        split_size = (self.config['SLIDE_ROWS'], self.config['SLIDE_COLS']) # split size e.g., (#height, #width) (rows, columns)
        
        self.H_points, self.H_size = self.sliding_window(image_height, split_size[0], self.config['SLIDE_OVERLAP'], self.config['SLIDE_FLAG']) 
        self.W_points, self.W_size = self.sliding_window(image_width, split_size[1], self.config['SLIDE_OVERLAP'], self.config['SLIDE_FLAG'])

        # GPU init
        self.image_gpu = cv2.cuda_GpuMat()

        # Hough Init
        self.image_canny = cv2.cuda_GpuMat()

        self.cannyFilter = cv2.cuda_CannyEdgeDetector(low_thresh=5, high_thresh=20, apperture_size=3)
        self.houghFilter = cv2.cuda_HoughLinesDetector(rho=1, theta=(np.pi/16), threshold=3, doSort=True, maxLines=32)

        self.houghResult_gpu = np.zeros(config['HOUGH_ANGLES'] * 2 * len(self.H_points) * len(self.W_points))

        # Law Mask Init
        self.create_lawMask(config['LAW_MASK'])

        # Optical Init
        self.image_prvs = cv2.cuda_GpuMat()
        self.image_next = cv2.cuda_GpuMat()
        self.nvof = cv2.cuda_FarnebackOpticalFlow.create(numLevels=3, pyrScale=0.5, fastPyramids=False, winSize=15,
                                                    numIters=3, polyN=5, polySigma=1.1, flags=0) 

        # Color feature Init
        self.hough_size = config['HOUGH_ANGLES'] * 2
        self.tensor_size = config['TENSOR_HISTBIN']
        self.law_size = len(config['LAW_MASK']) + 2
        self.flow_size = 5
        size_each_window = self.hough_size + self.tensor_size + self.law_size + self.flow_size

        self.feature_color_result = np.zeros(size_each_window * len(self.H_points) * len(self.W_points))
    
        # Depth feature Init
        self.depth_size = 1
        self.feature_depth_result = np.zeros(self.depth_size) * len(self.H_points) * len(self.W_points)

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

        # self.image_gpu.upload(image)
        # self.image_canny = self.cannyFilter.detect(self.image_gpu)
        # self.houghResult_gpu = self.houghFilter.detect(self.image_canny)

        # houghResult = self.houghResult_gpu.download()
        houghResult = np.zeros(30)
        
        return houghResult

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
        filter_size = self.config['TENSOR_BOXSIZE']
        Sxx = cv2.boxFilter(Sxx, cv2.CV_32F, (filter_size, filter_size))
        Syy = cv2.boxFilter(Syy, cv2.CV_32F, (filter_size, filter_size))
        Sxy = cv2.boxFilter(Sxy, cv2.CV_32F, (filter_size, filter_size))

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
        coherency = np.fmin(coherency, 1.0)
        coherency = np.fmax(coherency, 0.0)

        # Orientation angle
        orientation_Angle = cv2.phase(Sxx-Syy, 2.0*Sxy, angleInDegrees = True)
        orientation_Angle = 0.5 * orientation_Angle

        # Calculate Histogram
        # orientation_hist = cv2.calcHist(orientation_Angle, channels=[0], mask=None, histSize=[self.config['TENSOR_HISTBIN']], ranges=[0,180],  accumulate = False)
        # orientation_hist = cv2.normalize(orientation_hist, None, alpha=1, beta=None, norm_type=cv2.NORM_L1)
        index = 0
        histbin = self.config['TENSOR_HISTBIN']
        hist_result = np.zeros(histbin)
        for k in np.linspace(0.0, 180.0, histbin, endpoint=False):
            tmp1 = (orientation_Angle >= k) & (orientation_Angle < k + 180./histbin)
            if tmp1.any():
                hist_result[index] = np.mean(coherency[tmp1])
            index += 1

        return hist_result

    '''
    @ Law Mask
    '''
    def create_lawMask(self, masks):
        # Law's filters
        L3 = np.array([1,2,1])
        E3 = np.array([-1,0,1])
        S3 = np.array([-1,2,-1])
        L5 = np.array([1,4,6,4,1])
        E5 = np.array([-1,-2,0,2,1])
        S5 = np.array([-1,0,2,0,-1])
        W5 = np.array([-1,2,0,-2,1])
        R5 = np.array([1,-4,6,-4,1])

        # A dictionary of law masks
        # The most successful masks are {L5E5, E5S5, R5R5, L5S5, E5L5, S5E5, S5L5}
        lawMask_Dict = {
            "L5L5" : L5.reshape(5,1) * L5,
            "L5E5" : L5.reshape(5,1) * E5,
            "L5S5" : L5.reshape(5,1) * S5,
            "E5L5" : E5.reshape(5,1) * L5,
            "E5E5" : E5.reshape(5,1) * E5,
            "E5S5" : E5.reshape(5,1) * S5,
            "S5L5" : S5.reshape(5,1) * L5,
            "S5E5" : S5.reshape(5,1) * E5,
            "S5S5" : S5.reshape(5,1) * S5
        }

        self.law_masks = []
        for name in masks:
            self.law_masks.append((name, lawMask_Dict[name]))

    def law_feature(self, image):
        image = image.astype(np.float32)

        if len(image.shape) == 3:
            # Convert to YCrCb colorspace
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            
            # Apply Law's masks
            feature = np.zeros(len(self.law_masks) + 2)
            index = 0
            for name, mask in self.law_masks:
                if name == "L5L5":
                    for j in range(0,3):
                        image_filtered = cv2.filter2D(image[:,:,j], cv2.CV_32F, mask.astype(np.float32))
                        image_filtered = cv2.normalize(image_filtered, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        feature[index] = np.mean(abs(image_filtered))
                        index += 1            
                else:
                    image_filtered = cv2.filter2D(image[:,:,0], cv2.CV_32F, mask.astype(np.float32))
                    image_filtered = cv2.normalize(image_filtered, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    feature[index] = np.mean(abs(image_filtered))
                    index += 1    
        else:
            # if image is in grayscale

            # Apply Law's masks
            feature = np.zeros(len(mask))
            index = 0
            for name, mask in self.law_masks:
                image_filtered = cv2.filter2D(image[:,:,0], cv2.CV_32F, mask.astype(np.float32))
                image_filtered = cv2.normalize(image_filtered, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                feature[index] = np.mean(abs(image_filtered))
                index += 1    

        return feature 

    '''
    @ Optical Flow
    '''
    def flow_feature(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.image_prvs.empty():
            self.image_prvs.upload(image)
            return np.zeros(5)

        self.image_next.upload(image)

        # Call OF function
        flow_gpu = self.nvof.calc(self.image_prvs, self.image_next, None)
        flow = flow_gpu.download()

        self.image_prvs = self.image_next

        # Calculate magnitude and angle
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        flow_result = np.zeros(5, dtype=np.float32)
        flow_result[0] = np.amax(mag)
        flow_result[1] = np.amin(mag)
        flow_result[2] = (np.mean(flow[...,0]) + np.mean(flow[...,1])) / 2.0
        flow_result[3] = np.std(flow[...,0])
        flow_result[4] = np.std(flow[...,1])

        return flow_result

    '''
    @ Depth feature
    '''
    def depth_feature(self, image):
        # Reverse the value
        image = 255 - image
        
        # Currently just the average over the entire window
        feature = np.mean(np.abs(image)) / 255.

        return feature

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
                cropped_image = image[i:i+self.H_size, j:j+self.W_size]

                # Hough
                self.feature_color_result[k:k+self.hough_size] = self.hough_feature(cropped_image)
                k += self.hough_size
                # Structure Tensor
                self.feature_color_result[k:k+self.tensor_size] = self.tensor_feature(cropped_image)
                k += self.tensor_size
                # Law Mask
                self.feature_color_result[k:k+self.law_size] = self.law_feature(cropped_image)
                k += self.law_size
                # Optical Flow
                self.feature_color_result[k:k+self.flow_size] = self.flow_feature(cropped_image)
                k += self.flow_size

        return self.feature_color_result

    def update_depth(self, image):
        # Get features for each window
        k = 0
        for i in self.H_points:
            for j in self.W_points:              
                cropped_image = image[i:i+self.H_size, j:j+self.W_size]

                # Depth
                self.feature_depth_result[k:k+self.depth_size] = self.depth_feature(cropped_image)
                k += self.depth_size

        return self.feature_depth_result


    def step(self, image_color, image_depth):
        self.update_color(image_color)
        self.update_depth(image_depth)
        return np.concatenate((self.feature_color_result, self.feature_depth_result))


    def get_size(self):
        return len(self.feature_color_result) + len(self.feature_depth_result)


