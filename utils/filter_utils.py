import numpy as np

class SecondOrderLowPass():
    def __init__(self, cutoff_freq, sample_freq):
        self.a = np.empty((2,), dtype=np.float64)
        self.b = np.empty((2,), dtype=np.float64)
        self.i = np.empty((2,), dtype=np.float64)
        self.o = np.empty((2,), dtype=np.float64)

        self.configure(cutoff_freq, sample_freq)

    def configure(self, cutoff_freq, sample_freq):
        K = np.tan(np.pi * cutoff_freq / sample_freq)
        Q = 1. / (2. * np.cos(np.pi/4))
        poly = K*K + K/Q + 1.0
        self.a[0] = 2.0 * (K*K - 1.0) / poly
        self.a[1] = (K*K - K/Q + 1.0) / poly
        self.b[0] = K*K / poly
        self.b[1] = 2.0 * self.b[0]

    def update(self, value):
        if len(self.i) == 0:
            self.i[0], self.i[1], self.o[0], self.o[1] = value, value, value, value
            return value
        
        out = self.b[0] * value \
            + self.b[1] * self.i[0] \
            + self.b[0] * self.i[1] \
            - self.a[0] * self.o[0] \
            - self.a[1] * self.o[1]
        self.i[1] = self.i[0]
        self.i[0] = value
        self.o[1] = self.o[0]
        self.o[0] = out
        return out
        # return value
    
    def reset(self):
        self.i = np.empty((2,), dtype=np.float64)
        self.o = np.empty((2,), dtype=np.float64)