# Config Parameter
feature_config = {
    'SLIDE_ROWS': 4,        # slide row
    'SLIDE_COLS': 5,      # train_agent.run(output_dir=None, preload=True)      # slide column
    'SLIDE_OVERLAP': 0.0,   # overlap between two windows
    'SLIDE_FLAG': 1,        # flag=0 if provide the exact size, flag=1 if provide the number of windows
    
    'RADON_ANGLES': 15,     # number of radon angles

    'TENSOR_BOXSIZE': 5,    # boxfilter window size
    'TENSOR_HISTBIN': 15,   # number of histogram bins

    'LAW_MASK': [           # Law's masks
        "L5L5",
        "L5E5",
        "L5S5",
        "E5E5",
        "E5S5",
        "S5S5",
    ],          
}