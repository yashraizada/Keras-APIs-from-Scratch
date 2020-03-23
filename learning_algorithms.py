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
        
        self.update += self.momentum * self.update + self.learning_rate * gradient
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
        self.update += self.momentum * self.update + self.learning_rate * gradient_function(weight_look_ahead)
        return weight - self.update
