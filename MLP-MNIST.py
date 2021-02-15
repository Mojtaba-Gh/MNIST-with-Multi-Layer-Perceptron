import os
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from PIL import Image

class MLP:

	def __init__(self):
		self._alfa = 0.09 			# learning rate
		self._layers_config = [784, 512, 10]
		self._layer_count = len(self._layers_config)
		self._weights = []
		self._bias = []
		self.train_path = './Train_pics/'

		self.initial_weights()

	def int2array(self, x):
		# for change target
		target = [0 for x in range(self._layers_config[-1])]
		target[x] = 1
		return target

	def ReLU(self, x):
		return max(0, x)

	def ReLU_derivative(self, x):
		result = 1 if x > 0 else 0 
		return result

	def initial_weights(self):
		self._weights.append(None)
		self._bias.append(None)
		for i in range(1, self._layer_count ):
			pre_layer = self._layers_config[i-1]
			cur_layer = self._layers_config[i]
			layer_weights = np.random.normal(0, np.sqrt(2/pre_layer), (cur_layer, pre_layer))
			layer_bias = np.random.normal(0, np.sqrt(2/pre_layer), (cur_layer, 1))
			self._weights.append(layer_weights)
			self._bias.append(layer_bias)

	def feedforward(self, img):
		self.feed_neurons = [img]
		self.feed_neurons_in = [None]
		for i in range (1, self._layer_count):												# layers
			self.feed_neurons.append(np.zeros(self._layers_config[i]))
			self.feed_neurons_in.append(np.zeros(self._layers_config[i]))
			for j in range (0, self._layers_config[i]):										# layer length
				tmp = self._bias[i][j]
				for k in range(0, self._layers_config[i-1]):								# pre layer length
					tmp += self.feed_neurons[i-1][k] * self._weights[i][j][k]
				self.feed_neurons_in[i][j] = tmp
				self.feed_neurons[i][j] = self.ReLU(tmp)
		
	def backtrack(self, target):
		sigma_y = []								# for last layer
		delta_weight = self._weights.copy()
		delta_bias = self._bias.copy()

		for i in range (1, self._layer_count):
			delta_weight[i] = self._weights[i].copy()
			delta_bias[i] = self._bias[i].copy()


		# error in output layer 
		for k in range (self._layers_config[-1]):
			# Calculate sigma
			sigma_y.append( (target[k] - self.feed_neurons[-1][k]) * self.ReLU_derivative(self.feed_neurons_in[-1][k]) ) 
			# Calculate delta
			delta_bias[-1][k] = self._alfa * sigma_y[k]
			for j in range( self._layers_config[-2]):
				delta_weight[-1][k][j] = self._alfa * sigma_y[k] * self.feed_neurons[-2][j]
		
		# error in middle layer
		sigma_z = []
		for j in range ( self._layers_config[-2]):
			# Calculate sigma
			sigma_z.append(0)
			for k in range ( self._layers_config[-1]):
				sigma_z[j] += sigma_y[k] * self._weights[-1][k][j]
			sigma_z[j] = sigma_z[j] * self.ReLU_derivative(self.feed_neurons_in[-2][j])
			# Calculate delta
			delta_bias[-2][j] = self._alfa * sigma_z[j]
			for i in range ( self._layers_config[0]):
				delta_weight[-2][j][i] = self._alfa * sigma_z[j] * self.feed_neurons[0][i]

		# Calculate new wights
		for i in range ( 1, self._layer_count):
			for j in range( self._layers_config[i]):
				self._bias[i][j] += delta_bias[i][j]
				for k in range ( self._layers_config[i-1]):
					self._weights[i][j][k] += delta_weight[i][j][k]
		

	def learn(self):
		count = 0
		for input_path in os.listdir(self.train_path):
			count += 1
			print("Learn sample {}".format(count), end='\r')
			# get target from image neame.
			target = int(input_path.split('.')[0].split('-')[1])
			target = self.int2array(target)
			input_path = self.train_path + input_path
			
			# Read image
			img = Image.open(input_path ).convert('L')
			img = np.asarray_chkfinite(img)					# Convert image to numpy arrayy
			
			# Convert matrix(28 * 28) to vector (784)
			img = img.reshape(784)
			self.feedforward(img)							# self.feed_neurons will change
			self.backtrack(target)
			


a = MLP()
a.learn()