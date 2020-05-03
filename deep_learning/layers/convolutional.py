import numpy as np
from copy import copy
from initializers import * 
from activations import *
from optimizers import *
from utils import *

# Reference: https://keras.io/layers/convolutional; https://arxiv.org/pdf/1501.07338.pdf
class Conv():
	def __init__(self, rank, filters, kernel_size, input_shape=None, strides=(1, 1), padding='valid', data_format='channels_first', dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer=GlorotUniform(), bias_initializer=Zeros(), kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
		self.rank = rank
		self.filters = filters
		self.kernel_size = kernel_size
		self.strides = strides
		self.padding = padding
		self.data_format = data_format
		self.dilation_rate = dilation_rate
		self.activation = activation
		self.use_bias = use_bias
		self.kernel_initializer = kernel_initializer
		self.bias_initializer = bias_initializer
		self.kernel_regularizer = kernel_regularizer
		self.bias_regularizer = bias_regularizer
		self.activity_regularizer = activity_regularizer
		self.kernel_constraint = kernel_constraint
		self.bias_constraint = bias_constraint

		# Define learnable parameters
		self.kernel = None
		self.bias = None

		# Define dimensions
		self.input_shape = input_shape # Input Image Dim = Ch_In x Row_In x Col_in
		self.output_shape = None # Output Image Dim = Ch_Out x Row_Out x Col_Out
		self.kernel_shape = None # Kernel Dim = n_filters x (Row_Ker * Col_Ker * Ch_In)
		self.bias_shape = None # Kernel Dim = n_filters x 1

		self.layer_input = None
		self.batch_size = None

		# Cache
		self.linearized_layer_input = None
		self.linearized_pre_activation = None

	def add_weights(self, shape, data_format, initializer):
		if data_format == 'channels_first':
			return initializer(shape)

	def build(self):
		if self.data_format == 'channels_last':
			raise ValueError(str(self.data_format) + ' is not supported yet')

		# Update kernel & bias shape
		self.kernel_shape = (self.filters, self.kernel_size[0]*self.kernel_size[1]*self.input_shape[0])
		self.bias_shape = (self.filters, 1)

		self.kernel = self.add_weights(initializer=self.kernel_initializer, data_format=self.data_format, shape=self.kernel_shape)
		self.bias = self.add_weights(initializer=self.bias_initializer, data_format=self.data_format, shape=self.bias_shape)

	def image_to_column(self):
		i2c_matrix_dim = (self.batch_size, np.prod(self.output_shape[1:]), self.kernel_shape[1])
		i2c_matrix = np.zeros(i2c_matrix_dim)

		index = 0

		for row in range(0, self.input_shape[1], self.strides[0]):
			for col in range(0, self.input_shape[2], self.strides[1]):
				row_end, col_end = row+self.kernel_size[0], col+self.kernel_size[1]

				if (row_end<=self.input_shape[1]) and (col_end<=self.input_shape[2]):
					i2c_matrix[:, index, :] = straighten(self.layer_input[:, :, row:row_end, col:col_end])

					index += 1

		return i2c_matrix

	def column_to_image(self, input_matrix):
		c2i_matrix_dim = ((self.batch_size,) + (self.input_shape))
		c2i_matrix = np.zeros(c2i_matrix_dim)

		index = 0

		for row in range(0, self.input_shape[1], self.strides[0]):
			for col in range(0, self.input_shape[2], self.strides[1]):
				row_end, col_end = row+self.kernel_size[0], col+self.kernel_size[1]

				if (row_end<=self.input_shape[1]) and (col_end<=self.input_shape[2]):
					c2i_matrix[:, :, row:row_end, col:col_end] = unstraighten(input_matrix[:, index, :], (c2i_matrix_dim[1],) + self.kernel_size)
					index += 1

		return c2i_matrix

	def forward_prop_layer(self, layer_input, training):
		self.layer_input = layer_input
		self.batch_size = layer_input.shape[0]

		# retrieve column matrix & bridge to 2-D array
		self.linearized_layer_input = self.image_to_column().reshape(-1, self.kernel_shape[1])
		
		output = np.dot(self.linearized_layer_input, self.kernel.transpose())
		if self.use_bias:
			output = output + self.bias.transpose()
		if self.activation:
			output = self.activation(output)

		# cache & bridge to 3-D matrix
		self.linearized_pre_activation = output
		output = output.reshape(self.batch_size, np.prod(self.output_shape[1:]), self.filters)

		# convert the matrix back to operable format
		output = unstraighten(output, self.output_shape)

		return output

	def backward_prop_layer(self, prev_gradient, optimizer, learning_rate):
		cached_weights = self.kernel
		prev_gradient = prev_gradient.transpose(0, 2, 3, 1).reshape(-1, self.filters)
		if self.activation:
			prev_gradient = np.multiply(prev_gradient, self.activation.gradient(self.linearized_pre_activation))
			dk = np.dot(prev_gradient.transpose(), self.linearized_layer_input)
			db = np.sum(prev_gradient, axis = 0, keepdims = True)
			db = db.transpose()
		else:
			prev_gradient = np.multiply(prev_gradient, np.ones(self.linearized_pre_activation.shape))
			dk = np.dot(prev_gradient.transpose(), self.linearized_layer_input)
			db = np.sum(prev_gradient, axis = 0, keepdims = True)
			db = db.transpose()

		# creating 2 instances of optimizer - kernel, bias
		opt_kernel = copy(optimizer)
		opt_bias = copy(optimizer)

		self.kernel = opt_kernel.update(self.kernel, dk, learning_rate)
		self.bias = opt_bias.update(self.bias, db, learning_rate)

		prev_gradient = np.dot(prev_gradient, cached_weights)

		# bridge back to 3-D
		prev_gradient = prev_gradient.reshape(self.batch_size, np.prod(self.output_shape[1:]), prev_gradient.shape[-1])
		return self.column_to_image(prev_gradient)

	def set_input_shape(self, shape):
		if self.data_format == 'channels_first':
			self.input_shape = shape

	def compute_output_shape(self):
		# Define dim variables
		Ch_In, Row_In, Col_In = self.input_shape[0], self.input_shape[1], self.input_shape[2]
		Ch_Out, Row_Out, Col_Out = None, None, None 
		Row_Ker, Col_Ker = self.kernel_size[0], self.kernel_size[1]
		Row_Stri, Col_Stri = self.strides[0], self.strides[1]

		if self.padding == 'valid':
			Ch_Out = self.filters
			Row_Out = (Row_In - Row_Ker) // Row_Stri + 1
			Col_Out = (Col_In - Col_Ker) // Col_Stri + 1
		elif self.padding == 'same':
			Ch_Out = self.filters
			Row_Out = Row_In
			Col_Out = Col_In 
		else:
			raise ValueError('keyword: ' + str(self.padding) + ' not recognized')

		self.output_shape = (Ch_Out, Row_Out, Col_Out)

		if self.data_format == 'channels_first':
			return self.output_shape
