import numpy as np
from copy import copy
from initializers import * 
from activations import *
from optimizers import *

class Layers():
	def add_weights(self, shape, initializer):
		return initializer(shape)

class SimpleRNN(Layers):
	def __init__(self, units, input_shape=None, activation=TanH(), use_bias = True, kernel_initializer=GlorotUniform(), recurrent_initializer=Orthogonal(), bias_initializer=Zeros(), kernel_regularizer=None, return_sequences=False):
		self.units = units
		self.input_shape = input_shape
		self.activation = activation
		self.use_bias = use_bias
		self.kernel_initializer = kernel_initializer
		self.recurrent_initializer = recurrent_initializer
		self.bias_initializer = bias_initializer
		self.kernel_regularizer = kernel_regularizer
		self.return_sequences = return_sequences

		self.layer_input = None
		
		# Define internal shapes
		self.batch_size = None
		self.time_steps = None
		self.features = None

		# Set internal shapes if self.input_shape available
		if input_shape:
			self.set_internal_shapes(input_shape)

		# Define weight matrices
		self.U = None
		self.V = None
		self.W = None
		self.b = None
		self.c = None

		# Define state and output matrices
		self.state = None
		self.output = None

	def add_weights(self, shape, initializer):
		return super(Dense, self).add_weights(shape, initializer)

	def build(self):
		self.U = self.add_weights(initializer=self.kernel_initializer, shape=(self.units, self.features))
		self.V = self.add_weights(initializer=self.kernel_initializer, shape=(self.features, self.units))
		self.W = self.add_weights(initializer=self.kernel_initializer, shape=(self.units, self.units))
		self.b = self.add_weights(initializer=self.bias_initializer, shape=(self.units, 1))
		self.c = self.add_weights(initializer=self.bias_initializer, shape=(self.features, 1))

		self.state = self.add_weights(initializer=Zeros(), shape=(self.time_steps, self.units))
		self.state[0, :] = self.add_weights(initializer=recurrent_initializer, shape=(1, self.units))

		self.output = self.add_weights(initializer=Zeros(), shape=(self.time_steps, self.features))

	def forward_prop_layer(self, layer_input, training):
		self.layer_input = layer_input

		# Define batch_size
		self.batch_size = layer_input.shape[0]

		for time_step in range(1, time_steps+1):
			state_update = np.dot(self.U, self.layer_input[:, time_step, :].transpose()) + np.dot(self.W, self.state[time_step-1, :].transpose())

			if self.use_bias:
				state_update += self.b
			if self.activation:
				state_update = self.activation(state_update)

			output_update = np.dot(self.V, state_update)

			if self.use_bias:
				output_update += self.c
			if self.activation:
				output_update = self.activation(output_update)

			self.state[time_step, :] = state_update
			self.output[time_step, :] = output_update
		
		if self.return_sequences:
			return self.state
		
		return self.output

	def backward_prop_layer(self, prev_gradient, optimizer, learning_rate):
		pass

	def set_input_shape(self, shape):
		self.input_shape = shape
		self.set_internal_shapes(shape)

	def compute_output_shape(self):
		return self.units

	def set_internal_shapes(self, shape):
		self.time_steps = shape[0]
		self.features = shape[1]
