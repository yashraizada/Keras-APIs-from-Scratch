import numpy as np

class Metric():
	pass

class MAE(Metric):
	def __call__(self, y_hat, y):
		return np.mean(np.abs(y_hat - y))

	def __str__(self):
		return 'MAE'

class MSE(Metric):
	def __call__(self, y_hat, y):
		return np.mean(np.square(y_hat - y))

	def __str__(self):
		return 'MSE'

class RMSE(Metric):
	def __call__(self, y_hat, y):
		return np.sqrt(np.mean(np.square(y_hat - y)))

	def __str__(self):
		return 'RMSE'
