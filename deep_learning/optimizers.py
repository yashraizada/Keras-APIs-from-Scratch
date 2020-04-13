import numpy as np

# Reference: https://keras.io/optimizers
class Optimizer():
    def __init__(self, clipvalue=None, clipnorm=None):
        self.clipvalue = clipvalue
        self.clipnorm = clipnorm

    def value_clipper(self, gradients):
        return np.clip(-self.clipvalue, self.clipvalue, gradients)

# Stochastic Gradient Descent with/without Momentum
class SGD(Optimizer):
    def __init__(self, momentum = 0, clipvalue=None, clipnorm=None):
        super(SGD, self).__init__(clipvalue=clipvalue, clipnorm=clipnorm)
        self.momentum = momentum
        self.gamma = None

    def update(self, weight, gradient, learning_rate):
        if self.gamma is None:
            self.gamma = np.zeros(np.shape(weight))

        self.gamma = self.momentum * self.gamma + learning_rate * gradient
        
        if self.clipvalue:
            return self.value_clipper(weight - self.gamma)
        return weight - self.gamma

# Nesterov Accelerated Gradient Descent
class NAG(Optimizer):
    def __init__(self, momentum = 0.2, clipvalue=None, clipnorm=None):
        super(NAG, self).__init__(clipvalue=clipvalue, clipnorm=clipnorm)
        self.momentum = momentum
        self.gamma = None
        
    def update(self, weight, gradient_function, learning_rate):
        if self.gamma == None:
            self.gamma = np.zeros(np.shape(weight))
            
        weight_look_ahead = weight - self.momentum * self.gamma
        self.gamma = self.momentum * self.gamma + learning_rate * gradient_function(weight_look_ahead)
        
        if self.clipvalue:
            return self.value_clipper(weight - self.gamma)
        return weight - self.gamma

class AdaGrad(Optimizer):
    def __init__(self, clipvalue=None, clipnorm=None):
        super(AdaGrad, self).__init__(clipvalue=clipvalue, clipnorm=clipnorm)
        self.gradient_mag = None
        self.epsilon = 1e-10
    
    def update(self, weight, gradient, learning_rate):
        if self.gradient_mag == None:
            self.gradient_mag = np.zeros(np.shape(gradient))
        
        # accumulate magnitude of gradients
        self.gradient_mag += np.square(gradient)
        
        if self.clipvalue:
            return self.value_clipper(weight - learning_rate * gradient / np.sqrt(self.gradient_mag + self.epsilon))
        return weight - learning_rate * gradient / np.sqrt(self.gradient_mag + self.epsilon)

class RMSProp(Optimizer):
    def __init__(self, beta = 0.95, clipvalue=None, clipnorm=None):
        super(RMSProp, self).__init__(clipvalue=clipvalue, clipnorm=clipnorm)
        self.gradient_mag = None
        self.epsilon = 1e-5
        self.beta = beta
        
    def update(self, weight, gradient, learning_rate):
        if self.gradient_mag == None:
            self.gradient_mag = np.zeros(np.shape(gradient))
        
        # accumulate magnitude of gradients
        self.gradient_mag = self.beta * self.gradient_mag + (1 - self.beta) * np.square(gradient)
        
        if self.clipvalue:
            return self.value_clipper(weight - weight - learning_rate * gradient / np.sqrt(self.gradient_mag + self.epsilon))
        return weight - learning_rate * gradient / np.sqrt(self.gradient_mag + self.epsilon)

class Adam(Optimizer):
    def __init__(self, beta1 = 0.9, beta2 = 0.999, clipvalue=None, clipnorm=None):
        super(Adam, self).__init__(clipvalue=clipvalue, clipnorm=clipnorm)
        self.m = None
        self.v = None
        self.epsilon = 1e-5
        
        self.beta1 = beta1
        self.beta2 = beta2
        
    def update(self, weight, gradient, learning_rate):
        if self.m == None or self.v == None:
            self.m = np.zeros(np.shape(gradient))
            self.v = np.zeros(np.shape(gradient))
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(gradient)
        
        m_hat = self.m / (1 - self.beta1)
        v_hat = self.v / (1 - self.beta2)
        
        if self.clipvalue:
            return self.value_clipper(weight - learning_rate * m_hat / np.sqrt(v_hat + self.epsilon))
        return weight - learning_rate * m_hat / np.sqrt(v_hat + self.epsilon)
