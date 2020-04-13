import numpy as np

# Reference: https://keras.io/regularizers
class Regularizer():
	def __init__(self, l1=0., l2=0.):
		self.l1 = l1
		self.l2 = l2

	def __call__(self, x):
		regularization = 0.

		if self.l1:
			regularization += np.sum(np.abs(x))
		if self.l2:
			regularization += np.sum(np.square(x))

		return regularization

	def gradient(self, kernel):
		if self.l1:
			kernel[kernel>=0] = 1
			kernel[kernel<0] = -1
			return self.l1 * kernel
		if self.l2:
			return self.l2 * kernel



def l1(l=0.01):
	return Regularizer(l1=l)

def l2(l=0.01):
	return Regularizer(l2=l)

def l1l2(l1=0.01, l2=0.01):
	return Regularizer(l1=l1, l2=l2)
