import numpy as np
from copy import copy

from initializers import * 
from activations import *
from optimizers import *

# Reference: https://keras.io/layers/convolutional; https://arxiv.org/pdf/1501.07338.pdf
class Conv():
	def __init__(self, rank, filters, kernel_size, input_shape=None, strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer=GlorotUniform(), bias_initializer=Zeros(), kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
		self.rank = rank
		self.filters = filters
		self.kernel_size = kernel_size
		self.input_shape = input_shape
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

		# Input Image Dim = W1 x H1 x C1
		self.W1 = None
		self.H1 = None
		self.C1 = None

		# Output Image Dim = W2 x H2 x C2
		self.W2 = None
		self.H2 = None
		self.C2 = None

		# Kernel Dim = K_W x K_H x K_C
		self.K_W = self.kernel_size[0]
		self.K_H = self.kernel_size[1]
		self.K_C = None

		# Strides
		self.S_W = self.strides[0]
		self.S_H = self.strides[1]

	def build():
		if self.data_format == 'channels_first':
			self.C1 = self.input_shape[0]
			self.W1 = self.input_shape[1]
			self.H1 = self.input_shape[2]

		elif self.data_format == 'channels_last':
			self.W1 = self.input_shape[0]
			self.H1 = self.input_shape[1]
			self.C1 = self.input_shape[2]

		else:
			raise ValueError('keyword: ' + str(self.data_format) + ' not recognized')

		# Update kernel's channel
		self.K_C = self.C1

	def set_input_shape(self, shape):
		if self.data_format == 'channels_first':
			self.C1 = shape[0]
			self.W1 = shape[1]
			self.H1 = shape[2]

		if self.data_format == 'channels_last':
			self.W1 = shape[0]
			self.H1 = shape[1]
			self.C1 = shape[2]

	def compute_output_shape(self):
		if self.padding == 'valid':
			self.W2 = (self.W1 - self.K_W) // self.S_W + 1
			self.H2 = (self.H1 - self.K_H) // self.S_H + 1
		
		elif self.padding == 'same':
			self.W2 = self.W1
			self.H2 = self.H1

		else:
			raise ValueError('keyword: ' + str(self.padding) + ' not recognized')

		self.C2 = self.filters

		if self.data_format == 'channels_first':
			return (self.C2, self.W2, self.H2)
		if self.data_format == 'channels_last':
			return (self.W2, self.H2, self.C2)
