import numpy as np
import cv2

import config as cf

# Sliding window
# no extra

# Radon
from skimage.transform import radon
from skimage.util import img_as_float
import heapq

# Structure tensor
from structure_tensor import eig_special_2d, structure_tensor_2d

# Laws' Mask
# no extra

'''
Sliding Window
'''
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

'''
Radon Feature
'''
def radon_features(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = img_as_float(image)

    # Calculates the radon transform of a frame
    theta = np.linspace(0., 180., cf.RADON_ANGLES, endpoint=False)
    # Circle could = true or false
    sinogram = radon(image, theta=theta, circle = False, preserve_range=True)

    # Reducing matrix by a factor of 5 and averaging over values in between.
    sino_height, _ = sinogram.shape
    sinogram = (sinogram[(sino_height%5)::5]+sinogram[(sino_height%5)+1::5]+sinogram[(sino_height%5)+2::5]+sinogram[(sino_height%5)+3::5]+sinogram[(sino_height%5)+4::5])/5

    # Extracting weights by getting 2 largest values for each angle
    radon_weights = np.zeros((cf.RADON_ANGLES, 2))
    # Maybe could be done with enumerating, but its weird with ndarrays
    i = 0
    for row in sinogram.T:
        radon_weights[i] = heapq.nlargest(2, row)
        i = i+1

    # Scale the weights
    # !!! xmin and xmax may need to be changed depending on values of input image
    xmax = 512
    xmin = 0
    radon_weights = (radon_weights - xmin)/(xmax-xmin)

    # Fixing shape
    radon_weights = np.reshape(radon_weights, radon_weights.size)
    
    return radon_weights

'''
Structure Tensor Feature
'''
def tensor_feature(image, sigma, rho, filter_size, histBin, with_cuda=False):
    if not with_cuda:
        # Convert to grayscale if necessary
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Sober filter
        Sx = cv2.Sobel(image, cv2.CV_32F, 1, 0, 3)
        Sy = cv2.Sobel(image, cv2.CV_32F, 0, 1, 3)
        Sxx = cv2.multiply(Sx, Sx)
        Syy = cv2.multiply(Sy, Sy)
        Sxy = cv2.multiply(Sx, Sy)

        # Apply a box filter
        Sxx = cv2.boxFilter(Sxx, cv2.CV_32F, (filter_size, filter_size))
        Syy = cv2.boxFilter(Syy, cv2.CV_32F, (filter_size, filter_size))
        Sxy = cv2.boxFilter(Sxy, cv2.CV_32F, (filter_size, filter_size))

        # Calculate orientation angle
        orientation_Angle = cv2.phase(Sxx-Syy, 2.0*Sxy, angleInDegrees = True)
        orientation_Angle = 0.5 * orientation_Angle

        # Calculate Histogram
        orientation_hist = cv2.calcHist(orientation_Angle, channels=[0], mask=None, histSize=[histBin], ranges=[0,180],  accumulate = False)
        orientation_hist = cv2.normalize(orientation_hist, None, alpha=1, beta=None, norm_type=cv2.NORM_L1)

        return orientation_Angle, orientation_hist[..., 0]

    else:
        image = cv2.cuda_GpuMat(image)
        # Convert to grayscale if necessary
        if image.channels() == 3:
            image = cv2.cuda.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Sober filter
        filterX = cv2.cuda.createSobelFilter(image.depth(), cv2.CV_32F, dx=1, dy=0, ksize=3)
        filterY = cv2.cuda.createSobelFilter(image.depth(), cv2.CV_32F, dx=0, dy=1, ksize=3)
        Sx = filterX.apply(image)
        Sy = filterY.apply(image)
        Sxx = cv2.cuda.multiply(Sx, Sx)
        Syy = cv2.cuda.multiply(Sy, Sy)
        Sxy = cv2.cuda.multiply(Sx, Sy)

        boxfilter = cv2.cuda.createBoxFilter(cv2.CV_32F, cv2.CV_32F, ksize=(filter_size, filter_size))
        Sxx = boxfilter.apply(Sxx)
        Syy = boxfilter.apply(Syy)
        Sxy = boxfilter.apply(Sxy)

        # Calculate orientation angle
        orientation_Angle = cv2.cuda.phase(cv2.cuda.subtract(Sxx, Syy),
                                        cv2.cuda.multiply(Sxy, cv2.cuda_GpuMat(Sxy.size(), Sxy.type(), 2.0)),
                                        angleInDegrees=True)
        orientation_Angle = cv2.cuda.multiply(orientation_Angle, cv2.cuda_GpuMat(Sxy.size(), Sxy.type(), 0.5))

        # Calculate Histogram
        orientation_hist = cv2.calcHist([orientation_Angle.download()], channels=[0], mask=None, histSize=[histBin], ranges=[0,180], accumulate=False)
        orientation_hist = cv2.normalize(orientation_hist, None, alpha=1, beta=None, norm_type=cv2.NORM_L1)

        return orientation_Angle.download(), orientation_hist[..., 0]        

'''
Law Mask
'''
def create_lawMask():
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

    return lawMask_Dict    

def law_feature(image, masks, with_cuda=False):
    if not with_cuda:
        image = image.astype(np.float32)
        
        feature = np.zeros(cf.NUM_LAWS)

        image = cv2.cvtColor(image_gpu, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(image)

        #L5L5 Y
        image_filtered = masks[0].apply(Y)
        image_filtered = cv2.normalize(image_filtered, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        feature[0] = np.mean(abs(image_filtered))

        #L5L5 Cr
        image_filtered = masks[0].apply(Cr)
        image_filtered = cv2.normalize(image_filtered, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        feature[1] = np.mean(abs(image_filtered))

        #L5L5 Cb
        image_filtered = masks[0].apply(Cb)
        image_filtered = cv2.normalize(image_filtered, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        feature[2] = np.mean(abs(image_filtered))

        #L5E5 Y
        image_filtered = masks[1].apply(Y)
        image_filtered = cv2.normalize(image_filtered, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        feature[3] = np.mean(abs(image_filtered))

        #L5S5 Y
        image_filtered = masks[2].apply(Y)
        image_filtered = cv2.normalize(image_filtered, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        feature[4] = np.mean(abs(image_filtered))

        #E5E5 Y
        image_filtered = masks[3].apply(Y)
        image_filtered = cv2.normalize(image_filtered, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        feature[5] = np.mean(abs(image_filtered))

        #E5S5 Y
        image_filtered = masks[4].apply(Y)
        image_filtered = cv2.normalize(image_filtered, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        feature[6] = np.mean(abs(image_filtered))

        #S5S5 Y
        image_filtered = masks[5].apply(Y)
        image_filtered = cv2.normalize(image_filtered, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        feature[7] = np.mean(abs(image_filtered))

        return feature

    else:
        image = image.astype(np.float32)
        image_gpu = cv2.cuda_GpuMat(image)
        
        feature = np.zeros(cf.NUM_LAWS)

        image_gpu = cv2.cuda.cvtColor(image_gpu, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.cuda.split(image_gpu)

        #L5L5 Y
        image_filtered = masks[0].apply(Y)
        image_filtered = cv2.cuda.normalize(image_filtered, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        feature[0] = np.mean(abs(image_filtered.download()))

        #L5L5 Cr
        image_filtered = masks[0].apply(Cr)
        image_filtered = cv2.cuda.normalize(image_filtered, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        feature[1] = np.mean(abs(image_filtered.download()))

        #L5L5 Cb
        image_filtered = masks[0].apply(Cb)
        image_filtered = cv2.cuda.normalize(image_filtered, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        feature[2] = np.mean(abs(image_filtered.download()))

        #L5E5 Y
        image_filtered = masks[1].apply(Y)
        image_filtered = cv2.cuda.normalize(image_filtered, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        feature[3] = np.mean(abs(image_filtered.download()))

        #L5S5 Y
        image_filtered = masks[2].apply(Y)
        image_filtered = cv2.cuda.normalize(image_filtered, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        feature[4] = np.mean(abs(image_filtered.download()))

        #E5E5 Y
        image_filtered = masks[3].apply(Y)
        image_filtered = cv2.cuda.normalize(image_filtered, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        feature[5] = np.mean(abs(image_filtered.download()))

        #E5S5 Y
        image_filtered = masks[4].apply(Y)
        image_filtered = cv2.cuda.normalize(image_filtered, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        feature[6] = np.mean(abs(image_filtered.download()))

        #S5S5 Y
        image_filtered = masks[5].apply(Y)
        image_filtered = cv2.cuda.normalize(image_filtered, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        feature[7] = np.mean(abs(image_filtered.download()))

    return feature

'''
Optical Flow
'''
# seperate optical flow function for one frame
def opticalFlowOneFrame(frame1, frame2, with_cuda):
    if frame1.shape != frame2.shape:
        exit("Frame sizes do not match!")
    
    if not with_cuda:
        # Convert to grayscale
        frame_prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        frame_next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(frame_prvs, frame_next, None, 
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.1, flags=0)
    else:
        frame_prvs = cv2.cuda_GpuMat()
        frame_next = cv2.cuda_GpuMat()
        frame_prvs.upload(frame1)
        frame_next.upload(frame2)

        # Convert to grayscale
        frame_prvs = cv2.cuda.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        frame_next = cv2.cuda.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

        # Call OF function
        nvof = cv2.cuda_FarnebackOpticalFlow.create(numLevels=3, pyrScale=0.5, fastPyramids=False, winSize=15,
                                                    numIters=3, polyN=5, polySigma=1.1, flags=0) 
        flow_gpu = nvof.calc(frame1, frame2, None)
        flow = flow_gpu.download()

    # Calculate magnitude and angle
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    flow_result = np.zeros(cf.NUM_FLOWS, dtype=np.float32)
    flow_result[0] = np.amax(mag)         # maximum of the flow magnitude
    flow_result[1] = np.mean(flow[...,0]) # mean flow in x
    flow_result[2] = np.std(flow[...,0])  # standand deviation in x
    flow_result[3] = np.mean(flow[...,1]) # mean flow in y
    flow_result[4] = np.std(flow[...,1])  # standand deviation in y

    return flow_result

# Optical flow function for all windows and for overall, from color_features import opticalFlow
def opticalFlow(frame1, frame2, with_cuda=False):

    # Split the raw image into windows
    frame_height, frame_width, _ = frame1.shape
    split_size = (cf.COLOR_ROWS, cf.COLOR_COLS) # split size e.g., (#height, #width) (rows, columns)

    H_points, H_size = sliding_window(frame_height, split_size[0], cf.COLOR_OVERLAP, flag=1) # flag=0 if provide the exact size, flag=1 if provide the number of windows
    W_points, W_size = sliding_window(frame_width, split_size[1], cf.COLOR_OVERLAP, flag=1)

    # Get features for each window
    flow_features = np.zeros(cf.FLOW_FEATURES)
    k = 0
    for i in H_points:
        for j in W_points:
            cropped_frame1 = frame1[i:i+H_size, j:j+W_size]
            cropped_frame2 = frame1[i:i+H_size, j:j+W_size]
            flow_features[k:k+cf.NUM_FLOWS] = opticalFlowOneFrame(cropped_frame1, cropped_frame2, with_cuda)
            k += cf.FLOW_NUM_FLOWS

    flow_features[k:k+cf.NUM_FLOWS] = opticalFlowOneFrame(frame1, frame2)

    return flow_features

'''
Main Function
'''
def colorFeatures(image, law_masks):
    # Convert to BGR if not BGR
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Split the raw image into windows
    image_height, image_width, _ = image.shape
    split_size = (cf.COLOR_ROWS, cf.COLOR_COLS) # split size e.g., (#height, #width) (rows, columns)

    H_points, H_size = sliding_window(image_height, split_size[0], cf.COLOR_OVERLAP, flag=1) # flag=0 if provide the exact size, flag=1 if provide the number of windows
    W_points, W_size = sliding_window(image_width, split_size[1], cf.COLOR_OVERLAP, flag=1)

    # Get features for each window. 
    features = np.zeros(cf.COLOR_FEATURES_SIZE)
    k = 0
    for i in H_points:
        for j in W_points:
            cropped_image = image[i:i+H_size, j:j+W_size]
            # Radon
            features[k:k+(cf.RADON_ANGLES*2)] = radon_features(cropped_image)
            k += cf.RADON_ANGLES * 2
            # Structure Tensor
            _, features[k:k+cf.BINS] = tensor_feature(cropped_image, sigma=cf.SIGMA, rho=cf.RHO, filter_size=30, histBin=cf.BINS)
            k += cf.NUM_BINS
            # Law Mask
            features[k:k+cf.NUM_LAWS] = law_feature(cropped_image, masks=law_masks)
            k += cf.NUM_LAWS

    return features