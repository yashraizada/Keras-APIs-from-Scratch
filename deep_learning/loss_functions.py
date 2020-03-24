import numpy as np
from .activation_functions import Sigmoid

class LossFunction():
	def loss(self, y, y_hat):
		return NotImplementedError()

	def gradient(self, y, y_hat):
		return NotImplementedError()

class SquaredErrorLoss(LossFunction):
	def loss(self, y, y_hat):
		return np.square(y_hat - y) * 0.5

	def gradient(self, y, y_hat):
		return y_hat - y

class CrossEntropyLoss(LossFunction):
	def loss(self, y, y_hat):
		epsilon = 1e-10
		y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
		return -1 * (y * np.log(y_hat) + (1-y) * np.log(1-y_hat))

	def gradient(self, y, y_hat):
		return -(y / y_hat) + ((1-y) / (1-y_hat))
