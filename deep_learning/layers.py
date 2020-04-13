import numpy as np
from copy import copy
from initializers import * 
from activations import *
from optimizers import *

class Layers():
	def add_weights(self, shape, initializer):
		return initializer(shape)

class Dense(Layers):
	def __init__(self, units, input_shape=None, activation=None, use_bias = True, kernel_initializer=GlorotUniform(), bias_initializer=Zeros(), kernel_regularizer=None):
		self.units = units
		self.input_shape = input_shape
		self.activation = activation
		self.use_bias = use_bias
		self.kernel_initializer = kernel_initializer
		self.bias_initializer = bias_initializer
		self.kernel_regularizer = kernel_regularizer

		self.weighted_sum = None
		self.kernel = None
		self.bias = None
		self.batch_size = None

		# h_L-1
		self.layer_input = None
		# a_L
		self.pre_activation = None

	def add_weights(self, shape, initializer):
		return super(Dense, self).add_weights(shape, initializer)

	def build(self):
		self.kernel = self.add_weights(initializer=self.kernel_initializer, shape=(self.units, self.input_shape))
		self.bias = self.add_weights(initializer=self.bias_initializer, shape=(self.units, 1))

	def forward_prop_layer(self, layer_input, training):
		self.layer_input = layer_input
		self.batch_size = layer_input.shape[0]

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

		self.kernel = opt_weights.update(self.kernel, dk, learning_rate)
		self.bias = opt_bias.update(self.bias, db, learning_rate)

		if self.kernel_regularizer:
			self.kernel -= (self.kernel_regularizer.gradient(self.kernel)/self.batch_size)  

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

	def forward_prop_layer(self, layer_input, training):
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

# Reference: https://arxiv.org/pdf/1502.03167.pdf; https://kevinzakka.github.io/2016/09/14/batch_normalization; https://keras.io/layers/normalization
class BatchNormalization(Layers):
	def __init__(self, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer=Zeros(), gamma_initializer=Ones(), moving_mean_initializer=Zeros(), moving_variance_initializer=Ones()):
		self.axis = axis
		self.momentum = momentum
		self.epsilon = epsilon
		self.center = center
		self.scale = scale
		self.beta_initializer = beta_initializer
		self.gamma_initializer = gamma_initializer
		self.moving_mean_initializer = moving_mean_initializer
		self.moving_variance_initializer = moving_variance_initializer

		self.moving_mean = []
		self.moving_variance = []

		self.input_shape = None
		self.gamma = None
		self.beta = None
		self.batch_size = None

		# cache values
		self.layer_input = None
		self.x_hat = None
		self.inv_std_dev = None
	
	def add_weights(self, shape, initializer):
		return super(BatchNormalization, self).add_weights(shape, initializer)

	def build(self):
		self.gamma = self.add_weights(initializer=self.gamma_initializer, shape=(1, self.input_shape))
		self.beta = self.add_weights(initializer=self.beta_initializer, shape=(1, self.input_shape))

		self.moving_mean = self.add_weights(initializer=self.moving_mean_initializer, shape=(1, self.input_shape))
		self.moving_variance = self.add_weights(initializer=self.moving_variance_initializer, shape=(1, self.input_shape))

	def forward_prop_layer(self, layer_input, training):
		# update shape
		self.layer_input = layer_input
		self.batch_size = layer_input.shape[0]

		if training==True:
			# calculate values
			mean = np.mean(layer_input, axis=0)
			var = np.var(layer_input, axis=0)
			inv_std_dev = 1/np.sqrt(var + self.epsilon)

			# update moving_mean and moving_variance
			self.moving_mean = self.momentum * self.moving_mean + (1-self.momentum) * mean
			self.moving_variance = self.momentum * self.moving_variance + (1-self.momentum) * var

		else:
			mean = self.moving_mean
			var = self.moving_variance
			inv_std_dev = 1/np.sqrt(var + self.epsilon)

		x_hat = (layer_input - mean) * inv_std_dev

		if self.scale:
			x_hat = np.multiply(self.gamma, x_hat)
		if self.center:
			x_hat = x_hat + self.beta

		# cache
		self.x_hat = x_hat
		self.inv_std_dev = inv_std_dev

		return x_hat

	def backward_prop_layer(self, prev_gradient, optimizer, learning_rate):
		# dl_da wihtout any activation
		prev_gradient = np.multiply(prev_gradient, np.ones(self.layer_input.shape))

		dx_hat = np.sum(np.multiply(prev_gradient, self.gamma), axis=0, keepdims=True)
		dgamma = np.sum(np.multiply(prev_gradient, self.x_hat), axis=0, keepdims=True)
		dbeta = np.sum(prev_gradient, axis=0, keepdims=True)

		dx = (1./self.batch_size) * self.inv_std_dev * (self.batch_size*dx_hat - np.sum(dx_hat, axis=0) - np.multiply(self.x_hat, np.sum(np.multiply(dx_hat, self.x_hat), axis=0)))

		# creating 2 instances of optimizer - gamma, beta
		opt_gamma = copy(optimizer)
		opt_beta = copy(optimizer)

		self.gamma = opt_gamma.update(self.gamma, dgamma, learning_rate)
		self.beta = opt_beta.update(self.beta, dbeta, learning_rate)

	def set_input_shape(self, shape):
		self.input_shape = shape

	def compute_output_shape(self):
		return self.input_shape
