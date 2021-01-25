'''
Color Image
'''
# Color sliding windows
COLOR_ROWS = 10
COLOR_COLS = 15
COLOR_OVERLAP = 0.5

# Radon features
RADON_ANGLES = 15

# Structure tensor
TENSOR_SIGMA = 1.0     # A noise scale, structures smaller than sigma will be removed by smoothing.
TENSOR_RHO = 1.0       # An integration scale giving the size over the neighborhood in which the orientation is to be analysed.
TENSOR_NUM_BINS = 15   # Number of histogram bins

# Laws Masks

LAW_NUM_LAWS = 8

# Number of color features
COLOR_FEATURES_SIZE = (RADON_ANGLES*2 + TENSOR_NUM_BINS + LAW_NUM_LAWS)*COLOR_ROWS*COLOR_COLS

# Number of optical flow features
FLOW_NUM_FLOWS = 5
FLOW_FEATURES = (COLOR_ROWS * COLOR_COLS * FLOW_NUM_FLOWS) + FLOW_NUM_FLOWS

'''
Depth Image
'''
# Depth sliding window
DEPTH_ROWS = 10
DEPTH_COLS = 15
DEPTH_OVERLAP = 0.5

# Number of depth features
DEPTH_FEATURES_SIZE = DEPTH_ROWS * DEPTH_COLS
