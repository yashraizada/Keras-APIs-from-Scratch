import numpy as np

from layers import *
from loss_functions import *
from utils.data_manipulation import batch_generator
from utils.visualizations import plot_curves

class Model():
	pass

class Sequential(Model):
	def __init__(self):
		self.layers = []

		self.loss= None
		self.optimizer = None
		self.learning_rate = None

		# store errors and accuracy
		self.history = {'cost':[]}

		# store evaluation metrics
		self.evaluation_metrics = None
		
	def add(self, layer):
		if self.layers:
			shape = self.layers[-1].compute_output_shape()
			layer.set_input_shape(shape)

		self.layers.append(layer)

	def forward_propagation(self, X, training):
		for layer in self.layers:
			X = layer.forward_prop_layer(X, training)
		return X

	def backward_propagation(self, prev_gradient):
		for layer in reversed(self.layers):
			prev_gradient = layer.backward_prop_layer(prev_gradient, self.optimizer, self.learning_rate)

	def train_on_batch(self, X_batch, y_batch):
		# forward pass
		y_hat = self.forward_propagation(X_batch, training=True)

		# calculate cost
		cost = self.loss(y_batch, y_hat)
		cost = np.sum(cost) / X_batch.shape[0]

		# gradient wrt the Loss function
		loss_gradient = self.loss.gradient(y_batch, y_hat)

		# backward propagation step
		self.backward_propagation(prev_gradient=loss_gradient)

		return cost

	def evaluate_on_batch(self, X_batch, y_batch, batch_history, istrain):
		# forward pass
		y_hat = self.forward_propagation(X_batch, training=False)

		if istrain:
			prefix = 'train_'
		else:
			prefix = 'test_'

		for metric in self.evaluation_metrics:
			metric_result = metric(y_hat, y_batch)

			if prefix+str(metric) not in batch_history.keys():
				batch_history.update({prefix+str(metric): [metric_result]})
			else:
				batch_history[prefix+str(metric)].append(metric_result)

		return batch_history

	def compile(self, loss, optimizer, metrics=None):
		for layer in self.layers:
			if hasattr(layer, 'build'):
				layer.build()

		self.loss = loss
		self.optimizer = optimizer

		if metrics:
			self.evaluation_metrics = metrics

			# create train and test keys in the history dictionary
			for metric in metrics:
				self.history['train_' + str(metric)] = []
				self.history['test_' + str(metric)] = []

	def fit(self, X, y, batch_size=None, epochs=1, verbose=1, learning_rate=0.1, validation_split=0., validation_data=None, shuffle=True):
		
		self.learning_rate = learning_rate

		if shuffle:
			index_list = np.arange(X.shape[0])
			np.random.shuffle(index_list)
			X = X[index_list]
			y = y[index_list]

		if validation_data:
			X_val, y_val = validation_data
		elif validation_split and 0. < validation_split < 1:
			slice_index = int(X.shape[0] * (1-validation_split))
			X_val, y_val = X[slice_index:], y[slice_index:]
			X, y = X[:slice_index], y[:slice_index]

		if batch_size == None:
			batch_size = X.shape[0]

		# conversion to operable dimensions
		y = y.reshape(y.shape[0], 1)

		for epoch in range(epochs):
			print(f"Epoch: {epoch+1}")

			batch_history = {'cost':[]}

			for X_batch, y_batch in batch_generator(X, y, batch_size):
				cost = self.train_on_batch(X_batch, y_batch)

				# add cost to batch history
				batch_history['cost'].append(cost)

				if self.evaluation_metrics and (validation_data is not None or validation_split > 0.):
					batch_history = self.evaluate_on_batch(X_batch, y_batch, batch_history, istrain=True)
					batch_history = self.evaluate_on_batch(X_val, y_val, batch_history, istrain=False)

			for key in batch_history.keys():
				batch_history[key] = np.mean(batch_history[key]).round(2)
				self.history[key].append(batch_history[key])

			# print results
			print(batch_history)

	def plot_history(self, type_='loss'):
		plot_curves()(self.history, type_)
