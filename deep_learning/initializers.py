import numpy as np

# Keras Initializers  : https://keras.io/initializers

class Initializer():
    def __call__(self):
        return NotImplementedError

class Zeros(Initializer):
    def __init__(self, dtype = np.float):
        self.dtype = dtype
        
    def __call__(self, shape):
        return np.zeros(shape = shape, dtype = self.dtype)
    
class Ones(Initializer):
    def __init__(self, dtype = np.float):
        self.dtype = dtype
        
    def __call__(self, shape):
        return np.ones(shape = shape, dtype = self.dtype)
    
class Constant(Initializer):
    def __init__(self, constant, dtype = np.float):
        self.constant = constant
        self.dtype = dtype
        
    def __call__(self, shape):
        return np.full(shape = shape, fill_value = self.constant, dtype = self.dtype)
    
class RandomNormal(Initializer):
    def __init__(self, mean=0.0, stddev=0.05):
        self.mean = mean
        self.stddev = stddev
        
    def __call__(self, shape):
        return np.random.normal(size = shape, loc=self.mean, scale=self.stddev)
    
class RandomUniform(Initializer):
    def __init__(self, minval=-0.05, maxval=0.05):
        self.minval = minval
        self.maxval = maxval
        
    def __call__(self, shape):
        return np.random.uniform(size = shape, low=self.minval, high=self.maxval)

# Glorot Normal (not truncated)
class GlorotNormal(Initializer):
    def __call__(self, shape):
        stddev = np.sqrt(2/(shape[0] + shape[1]))
        return np.random.normal(size = shape, loc=0, scale=stddev)
    
class GlorotUniform(Initializer):
    def __call__(self, shape):
        limit = np.sqrt(6/(shape[0] + shape[1]))
        return np.random.uniform(size = shape, low=-limit, high=limit)
