import pandas as pd
import numpy as np

filename = 'airsim.csv'

data = pd.read_csv(filename, dtype=np.float32)
# print(dir(data))
print(data.to_numpy()[:,0].shape)