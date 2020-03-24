import numpy as np

# Stochastic Gradient Descent with/without Momentum
class SDG():
    def __init__(self, learning_rate = 0.001, momentum = 0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.update = None
    
    def update(self, weight, gradient):
        if self.update == None:
            self.update = np.zeros(np.shape(weight))
        
        self.update = self.momentum * self.update + self.learning_rate * gradient
        return weight - self.update

# Nesterov Accelerated Gradient Descent
class NAG():
    def __init__(self, learning_rate = 0.001, momentum = 0.2):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.update = []
        
    def update(self, weight, gradient_function):
        if self.update == None:
            self.update = np.zeros(np.shape(weight))
            
        weight_look_ahead = weight - self.momentum * self.update
        self.update = self.momentum * self.update + self.learning_rate * gradient_function(weight_look_ahead)
        return weight - self.update

class AdaGrad():
    def __init__(self, learning_rate = 0.001):
        self.learning_rate = learning_rate
        self.gradient_mag = None
        self.epsilon = 1e-10
    
    def update(self, weight, gradient):
        if self.gradient_mag == None:
            self.gradient_mag = np.zeros(np.shape(gradient))
        
        # accumulate magnitude of gradients
        self.gradient_mag += np.square(gradient)
        
        return weight - self.learning_rate * gradient / np.sqrt(self.gradient_mag + self.epsilon)

class RMSProp():
    def __init__(self, learning_rate = 0.001, beta = 0.95):
        self.learning_rate = learning_rate
        self.gradient_mag = None
        self.epsilon = 1e-5
        self.beta = beta
        
    def update(self, weight, gradient):
        if self.gradient_mag == None:
            self.gradient_mag = np.zeros(np.shape(gradient))
        
        # accumulate magnitude of gradients
        self.gradient_mag = self.beta * self.gradient_mag + (1 - self.beta) * np.square(gradient)
        
        return weight - self.learning_rate * gradient / np.sqrt(self.gradient_mag + self.epsilon)

class Adam():
    def __init__(self, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999):
        self.learning_rate = learning_rate
        
        self.m = None
        self.v = None
        self.epsilon = 1e-5
        
        self.beta1 = beta1
        self.beta2 = beta2
        
    def update(self, weight, gradient):
        if self.m == None or self.v == None:
            self.m = np.zeros(np.shape(gradient))
            self.v = np.zeros(np.shape(gradient))
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.sqaure(gradient)
        
        m_hat = self.m / (1 - self.beta1)
        v_hat = self.v / (1 - self.beta2)
        
        return weight - self.learning_rate * m_hat / np.sqrt(v_hat + self.epsilon)
