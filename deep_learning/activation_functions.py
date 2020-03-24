import numpy as np

class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))

class TanH():
    def __call__(self, x):
        return np.tanh(x)

    def gradient(self, x):
        return 1 - np.power(self.__call__(x), 2)

class ReLU():
    def __call__(self, x):
        return x if x>0 else 0

    def gradient(self, x):
        return 1 if x>0 else 0

class LeakyReLU():
    def __init__(self, constant = 0.01):
        self.constant = constant
        
    def __call__(self, x):
        return x if x>0 else x*self.constant

    def gradient(self, x):
        return 1 if x>0 else self.constant

class Softmax():
    def __call__(self, x):
        return np.exp(x) / np.sum(np.exp(x), keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)
