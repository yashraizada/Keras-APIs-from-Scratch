import numpy as np

# Keras Activations : https://keras.io/activations and https://keras.io/layers/advanced-activations

class Activation():
    def __call__(self, x):
        return NotImplementedError

    def gradient(self, x):
        return NotImplementedError

class ELU(Activation):
    def __call__(self, x, alpha=1.0, scale=1.0):
        return scale*x if x>=0 else scale*alpha*(np.exp(x)-1)

    def gradient(self, x):
        return scale if x>=0 else scale*alpha*np.exp(x)

class Softplus(Activation):
    def __call__(self, x):
        return np.log(1 + np.exp(x))

    def gradient(self, x):
        return 1/(1 + np.exp(-x))

class Sigmoid(Activation):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))

class TanH(Activation):
    def __call__(self, x):
        return np.tanh(x)

    def gradient(self, x):
        return 1 - np.power(self.__call__(x), 2)

class ReLU(Activation):
    def __call__(self, x):
        return x if x>0 else 0

    def gradient(self, x):
        return 1 if x>0 else 0

class LeakyReLU(Activation):
    def __init__(self, constant = 0.01):
        self.constant = constant
        
    def __call__(self, x):
        return x if x>0 else x*self.constant

    def gradient(self, x):
        return 1 if x>0 else self.constant

class Softmax(Activation):
    def __call__(self, x):
        return np.exp(x) / np.sum(np.exp(x), keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)
