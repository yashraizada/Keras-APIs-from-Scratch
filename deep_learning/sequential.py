import numpy as np
from tqdm import tqdm

from layers import *
from loss_functions import *
from utils.data_manipulation import batch_generator

class Model():
	pass

class Sequential(Model):
	def __init__(self):
		self.layers = []

		self.loss= None
		self.optimizer = None
		self.learning_rate = None
		
	def add(self, layer):
		if self.layers:
			shape = self.layers[-1].compute_output_shape()
			layer.set_input_shape(shape)

		self.layers.append(layer)

	def forward_propagation(self, X):
		for layer in self.layers:
			X = layer.forward_prop_layer(X)
		return X

	def backward_propagation(self, prev_gradient):
		for layer in reversed(self.layers):
			layer.backward_prop_layer(prev_gradient, self.optimizer, self.learning_rate)

	def train_on_batch(self, X_batch, y_batch):
		# forward pass
		y_hat = self.forward_propagation(X_batch)

		# calculate cost
		cost = self.loss(y_batch, y_hat)
		cost = np.sum(cost) / X_batch.shape[0]

		# gradient wrt the Loss function
		loss_gradient = self.loss.gradient(y_batch, y_hat)

		# backward propagation step
		self.backward_propagation(prev_gradient=loss_gradient)

	def compile(self, loss, optimizer):
		for layer in self.layers:
			layer.build()

		self.loss = loss
		self.optimizer = optimizer

	def fit(self, X, y, batch_size=None, epochs=1, verbose=1, learning_rate=0.01, validation_split=0., validation_data=None, shuffle=True):
		
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

		if batch_size == None:
			batch_size = X.shape[0]

		y = y.reshape(y.shape[0], 1)

		for _ in tqdm(range(epochs)):
			for X_batch, y_batch in batch_generator(X, y, batch_size):
				self.train_on_batch(X_batch, y_batch)
