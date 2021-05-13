import numpy as np
import cv2
import time

class FeatureExtract():
    def __init__(self, config, image_size, printout=False):
        self.config = config
        self.image_size = image_size
        self.printout = printout
        # Configure
        self.configure()

    def configure(self):
        # Sliding Window Init (only executed once)
        image_height = self.image_size[0]
        image_width = self.image_size[1]
        split_size = (self.config['SLIDE_ROWS'], self.config['SLIDE_COLS']) # split size e.g., (#height, #width) (rows, columns)
        
        self.H_points, self.H_size = self.sliding_window(image_height, split_size[0], self.config['SLIDE_OVERLAP'], self.config['SLIDE_FLAG']) 
        self.W_points, self.W_size = self.sliding_window(image_width, split_size[1], self.config['SLIDE_OVERLAP'], self.config['SLIDE_FLAG'])
        
        # Images in GPU
        self.image_next_bgr = cv2.cuda_GpuMat((self.image_size[1], self.image_size[0]), cv2.CV_8UC3)
        self.image_next_gray = cv2.cuda_GpuMat((self.image_size[1], self.image_size[0]), cv2.CV_8U)

        # Feature List
        self.feature_list = [
            # Name,              Size,                           Function handle,      Init Function handle,
            ('hough',            self.config['HOUGH_ANGLES'],    self.hough_apply,   self.hough_init),
            ('structure_tensor', self.config['TENSOR_HISTBIN'],  self.tensor_apply,  self.tensor_init),
            ('law_mask',         len(self.config['LAW_MASK']),   self.law_apply,     self.law_init),
            ('optical_flow',     3,                              self.flow_apply,    self.flow_init),
            ('depth',            1,                              self.depth_apply,   self.depth_init),
        ]

        self.size_each_window = 0
        self.time_dict = {}
        for name, size, _, init_function in self.feature_list:
            self.size_each_window += size
            self.time_dict[name] = 0.0
            init_function()
        self.feature_result = np.zeros(self.size_each_window * len(self.H_points) * len(self.W_points))

    """
    @ Sliding Window
    """
    @staticmethod
    def sliding_window(size, split_size, overlap=0.0, flag=0):
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

    """
    @ Hough Feature
    """
    def hough_init(self):
        self.image_canny = cv2.cuda_GpuMat((self.W_size, self.H_size), cv2.CV_8U)
        self.cannyFilter = cv2.cuda.createCannyEdgeDetector(low_thresh=5, high_thresh=20, apperture_size=3)
        self.houghFilter = cv2.cuda.createHoughLinesDetector(rho=1, theta=(np.pi/60), threshold=3, doSort=True, maxLines=30)
        self.houghResult_gpu = cv2.cuda_GpuMat((30, 2), cv2.CV_32FC2)

    def hough_apply(self):
        # Result
        hough_result = np.zeros(self.config['HOUGH_ANGLES'] * len(self.H_points) * len(self.W_points))
        index = 0
        scale = 1
        for i in self.H_points:
            for j in self.W_points:
                self.cannyFilter.detect(self.image_next_gray.rowRange(i,i+self.H_size).colRange(j,j+self.W_size), self.image_canny)
                self.houghFilter.detect(self.image_canny, self.houghResult_gpu)
                houghLines = self.houghResult_gpu.download()

                if houghLines is not None:
                    for k in range(0, len(houghLines[0])):
                        thetaIndex = int(houghLines[0][k][1] / (np.pi/15))
                        hough_result[thetaIndex + self.config['HOUGH_ANGLES']*index] += 1/32
                index += 1

        return hough_result / scale

    """
    @ Structure Tensor
    """
    def tensor_init(self):
        pass

    def tensor_apply(self, image):
         # Convert to grayscale
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

        # Result
        histbin = self.config['TENSOR_HISTBIN']
        tensor_result = np.zeros(histbin * len(self.H_points) * len(self.W_points))
        index = 0
        scale = 255 # a scale to normalize the output
        for i in self.H_points:
            for j in self.W_points:
                for k in np.linspace(0.0, 180.0, histbin, endpoint=False):
                    tmp1 = (orientation_Angle[i:i+self.H_size, j:j+self.W_size] >= k) & (orientation_Angle[i:i+self.H_size, j:j+self.W_size] < k + 180./histbin)
                    if tmp1.any():
                        tensor_result[index] = np.sum(coherency[i:i+self.H_size, j:j+self.W_size][tmp1]) 
                    index += 1

        return tensor_result / scale

    """
    @ Law Mask
    """
    def law_init(self):
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
        # Note: The most successful masks are {L5E5, E5S5, R5R5, L5S5, E5L5, S5E5, S5L5}
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

        self.image_next_bgr32 = cv2.cuda_GpuMat((self.image_size[1], self.image_size[0]), cv2.CV_32FC3)
        self.image_next_ycrcb = cv2.cuda_GpuMat((self.image_size[1], self.image_size[0]), cv2.CV_32FC3)
        Y, _, _ = cv2.cuda.split(self.image_next_ycrcb) # To speed up the first iteration
        self.law_masks_gpu = []
        for name in self.config['LAW_MASK']:
            self.law_masks_gpu.append((name, cv2.cuda.createLinearFilter(cv2.CV_32F, cv2.CV_32F, kernel=lawMask_Dict[name])))
            self.law_masks_gpu[-1][1].apply(Y) # To speed up the first iteration
        
    def law_apply(self):
        # Convert to YCrCb colorspace
        self.image_next_bgr32 = self.image_next_bgr.convertTo(cv2.CV_32FC3, self.image_next_bgr32)
        self.image_next_ycrcb = cv2.cuda.cvtColor(self.image_next_bgr32, cv2.COLOR_BGR2YCrCb)
        Y, _, _ = cv2.cuda.split(self.image_next_ycrcb)
        
        # Apply Law's masks
        image_filtered = {}
        for name, mask in self.law_masks_gpu:
            image_filtered[name] = mask.apply(Y).download()

        # Result
        law_result = np.zeros(len(self.law_masks_gpu) * len(self.H_points) * len(self.W_points))
        index = 0
        scale = 255 # a scale to normalize the output
        for i in self.H_points:
            for j in self.W_points:
                for name, mask in self.law_masks_gpu:
                    law_result[index] = np.mean(abs(image_filtered[name][i:i+self.H_size, j:j+self.W_size]))
                    index += 1  
                
        return law_result / scale

    """
    @ Optical Flow
    """
    def flow_init(self):
        self.is_first_image = True 
        self.image_prvs_gray = cv2.cuda_GpuMat((self.image_size[1], self.image_size[0]), cv2.CV_8U)
        self.nvof = cv2.cuda_FarnebackOpticalFlow.create(numLevels=3, pyrScale=0.5, fastPyramids=False, winSize=15,
                                                    numIters=3, polyN=5, polySigma=1.1, flags=0) 
        self.flow_gpu = cv2.cuda_GpuMat((self.image_size[1], self.image_size[0]), cv2.CV_32FC2)
        self.nvof.calc(self.image_prvs_gray, self.image_next_gray, None) # only to speed up the first calculation
    
    def flow_apply(self):
        # Call OF function
        self.nvof.calc(self.image_prvs_gray, self.image_next_gray, self.flow_gpu, None)
        flow = self.flow_gpu.download()

        # Calculate magnitude and angle
        mag = np.sqrt(np.square(flow[..., 0]) + np.square(flow[..., 1]))
        
        # Result
        flow_result = np.zeros(3 * len(self.H_points) * len(self.W_points))
        index = 0
        scale = 10
        for i in self.H_points:
            for j in self.W_points:
                flow_result[index] = np.amax(mag[i:i+self.H_size, j:j+self.W_size])
                flow_result[index+1] = np.amin(mag[i:i+self.H_size, j:j+self.W_size])
                flow_result[index+2] = np.mean(mag[i:i+self.H_size, j:j+self.W_size])
                index += 3

        return flow_result / scale

    """
    @ Depth feature
    """
    def depth_init(self):
        pass

    def depth_apply(self, image, reverse=True):
        # Reverse the pixel value
        if reverse:
            image = 255 - image

        # Result
        depth_result = np.zeros(1 * len(self.H_points) * len(self.W_points))
        index = 0
        for i in self.H_points:
            for j in self.W_points:
                # Currently just the average over the entire window
                depth_result[index] = np.mean(image[i:i+self.H_size, j:j+self.W_size]) / 255. # in range [0.0, 1.0]
                index += 1

        return depth_result

    """
    @ Step function
    """
    def step(self, image_color, image_depth, color_space='RGB'):   
        if color_space is 'RGB':
            # Convert from RGB to BGR
            img_color = cv2.cvtColor(image_color, cv2.COLOR_RGB2BGR)
        else:
            img_color = image_color.copy()
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        # Update image in GPU
        self.image_next_bgr.upload(img_color)
        self.image_next_gray.upload(img_gray)

        # for optical flow
        if self.is_first_image:
            self.image_prvs_gray.upload(img_gray)
            self.is_first_image = False
        
        # Get features for each window
        k = 0
        for name, size, function, _ in self.feature_list:
            all_size = size * len(self.H_points) * len(self.W_points)
            start_time = time.perf_counter()
            if name is 'depth':
                self.feature_result[k:k+all_size] = function(image_depth)
            elif name is 'structure_tensor':
                self.feature_result[k:k+all_size] = function(img_color)
            else:
                self.feature_result[k:k+all_size] = function()
            self.time_dict[name] = time.perf_counter() - start_time 
            k += all_size
    
        # for optical flow
        self.image_next_gray.copyTo(self.image_prvs_gray)

        # Update time for each feature functions
        self.format_time(self.printout)
        
        return self.feature_result

    """
    Auxiliary Functions
    """
    def format_time(self, printout=True):
        if printout:
            # Format the elapsed time and print out 
            factor = 1.0 / sum(self.time_dict.values())
            for key in self.time_dict:
                self.time_dict[key] *= factor
                print('{:s}: {:.2%},'.format(key, self.time_dict[key]), end=" ")        
            print('')

    def reset(self):
        self.is_first_image = True
        self.feature_result.fill(0.0)

    @staticmethod
    def get_size(config, image_size):
        height, width = image_size
        split_size = (config['SLIDE_ROWS'], config['SLIDE_COLS']) # split size e.g., (#height, #width) (rows, columns)
        
        H_points, _ = FeatureExtract.sliding_window(height, split_size[0], config['SLIDE_OVERLAP'], config['SLIDE_FLAG']) 
        W_points, _ = FeatureExtract.sliding_window(width, split_size[1], config['SLIDE_OVERLAP'], config['SLIDE_FLAG'])
        
        # Feature List
        feature_list = [
            # Name,              Size,
            ('hough',            config['HOUGH_ANGLES']),
            ('structure_tensor', config['TENSOR_HISTBIN']),
            ('law_mask',         len(config['LAW_MASK'])),
            ('optical_flow',     3),
            ('depth',            1),
        ]

        size_each_window = 0
        for _, size in feature_list:
            size_each_window += size

        return size_each_window * len(H_points) * len(W_points)