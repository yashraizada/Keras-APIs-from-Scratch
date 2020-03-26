import numpy as np

class Constant():
    def __init__(self, constant, shape, dtype = np.float):
        self.constant = constant
        self.shape = shape
        self.dtype = dtype
        
    def __call__(self):
        return np.full(shape = self.shape, fill_value = self.constant, dtype = self.dtype)
    
class RandomNormal():
    def __init__(self, size, mean=0.0, stddev=0.05):
        self.mean = mean
        self.stddev = stddev
        self.shape = size
        
    def __call__(self):
        return np.random.normal(size = self.shape, loc=self.mean, scale=self.stddev)
    
class RandomUniform():
    def __init__(self, size, minval=-0.05, maxval=0.05):
        self.minval = minval
        self.maxval = maxval
        self.shape = size
        
    def __call__(self):
        return np.random.uniform(size = self.shape, low=self.minval, high=self.maxval)
