import numpy as np

class Metric():
	pass

class MAE(Metric):
	def __init__(self, y_hat, y):
		self.y_hat = y_hat
		self.y = y_hat

	def __call__(self):
		return np.mean(np.abs(y_hat - y))

	def __str__(self):
		return 'MAE'

class MSE(Metric):
	def __init__(self, y_hat, y):
		self.y_hat = y_hat
		self.y = y_hat

	def __call__(self):
		return np.mean(np.square(y_hat - y))

	def __str__(self):
		return 'MSE'

class RMSE(Metric):
	def __init__(self, y_hat, y):
		self.y_hat = y_hat
		self.y = y_hat

	def __call__(self):
		return np.sqrt(np.mean(np.square(y_hat - y)))

	def __str__(self):
		return 'RMSE'
