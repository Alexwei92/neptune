# Config Parameter
feature_config = {
    'SLIDE_ROWS': 6,        # slide row
    'SLIDE_COLS': 8,        # slide column
    'SLIDE_OVERLAP': 0.0,   # overlap between two windows
    'SLIDE_FLAG': 1,        # flag=0 if provide the exact size, flag=1 if provide the number of windows
    
    #'HOUGH_ANGLES': 15,     # number of hough angles
    'HOUGH_ANGLES': 15,     # number of hough angles

    'TENSOR_FILTSIZE': 5,   # boxfilter/GaussianBlur filter size
    'TENSOR_HISTBIN': 15,   # number of histogram bins

    'LAW_MASK': [           # Law's masks
        "L5E5",
        "L5S5",
        "L5R5",
        "E5E5",
        "E5S5",
        "E5R5",
        "S5S5",
        "S5R5",
        "R5R5",
    ],

    'CMD_NPRVS': 5,         # number of previous commands
    'CMD_DECAY': 0.8        # exponential time decaying constant
}