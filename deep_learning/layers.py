import numpy as np
from copy import copy
from initializers import * 
from activations import *
from optimizers import *

class Layers():
	def add_weights(self, shape, initializer):
		return initializer(shape)

class Dense(Layers):
	def __init__(self, units, input_shape=None, activation=None, use_bias = True, kernel_initializer=GlorotUniform(), bias_initializer=Zeros()):
		self.units = units
		self.input_shape = input_shape
		self.activation = activation
		self.use_bias = use_bias
		self.kernel_initializer = kernel_initializer
		self.bias_initializer = bias_initializer
		# self.kernel_regularizer = kernel_regularizer
		# self.bias_regularizer = bias_regularizer

		self.weighted_sum = None
		self.kernel = None
		self.bias = None

		# h_L-1
		self.layer_input = None
		# a_L
		self.pre_activation = None

	def add_weights(self, shape, initializer):
		return super(Dense, self).add_weights(shape, initializer)

	def build(self):
		self.kernel = self.add_weights(initializer=self.kernel_initializer, shape=(self.units, self.input_shape))
		self.bias = self.add_weights(initializer=self.bias_initializer, shape=(self.units, 1))

	def forward_prop_layer(self, layer_input):
		self.layer_input = layer_input
		output = np.dot(self.layer_input, self.kernel.transpose())

		if self.use_bias:
			output = output + self.bias.transpose()

		self.pre_activation = output
		
		if self.activation:
			output = self.activation(output)

		return output

	def backward_prop_layer(self, prev_gradient, optimizer, learning_rate):
		cached_weights = self.kernel

		if self.activation:
			prev_gradient = np.multiply(prev_gradient, self.activation.gradient(self.pre_activation))
			dk = np.dot(prev_gradient.transpose(), self.layer_input)
			db = np.sum(prev_gradient, axis = 0, keepdims = True)
			db = db.transpose()
		else:
			prev_gradient = np.multiply(prev_gradient, np.ones(self.pre_activation.shape))
			dk = np.dot(prev_gradient.transpose(), self.layer_input)
			db = np.sum(prev_gradient, axis = 0, keepdims = True)
			db = db.transpose()

		# creating 2 instances of optimizer - weights, bias
		opt_weights = copy(optimizer)
		opt_bias = copy(optimizer)

		self.kernel = opt_weights.update(self.kernel, dk)
		self.bias = opt_bias.update(self.bias, db)

		return np.dot(prev_gradient, cached_weights)

	def set_input_shape(self, shape):
		self.input_shape = shape

	def compute_output_shape(self):
		return self.units


# Reference: http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf; https://keras.io/layers/core
class Dropout(Layers):
	def __init__(self, rate, noise_shape=None):
		self.rate = np.clip(rate, 0., 1.)
		self.noise_shape = noise_shape
		self.layer_input_shape = None
		self.bernoulli_mask = None

		self.input_shape = None

	def get_noise_shape(self):
		if self.noise_shape == None:
			return self.layer_input_shape

		return self.noise_shape

	def forward_prop_layer(self, layer_input):
		if 0. < self.rate < 1.:
			self.layer_input_shape = layer_input.shape
			self.bernoulli_mask = np.random.choice([0., 1.], self.get_noise_shape(), replace = True, p=[self.rate, 1-self.rate]) / (1-self.rate)
			
			return np.multiply(layer_input, self.bernoulli_mask)

		return layer_input

	def backward_prop_layer(self, prev_gradient, optimizer, learning_rate):
		if 0. < self.rate < 1.:
			return np.multiply(prev_gradient, self.bernoulli_mask) / (1-self.rate)

		return prev_gradient

	def set_input_shape(self, shape):
		self.input_shape = shape

	def compute_output_shape(self):
		return self.input_shape
