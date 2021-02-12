import os
import numpy as np
import matplotlib as mpl

class MLP:
	# class neuron:
	# 	def __init__(self, weights):
	# 		self._bias = 0
	# 		self._weight = [0.1 for i in range(weights)]
	# 		# single_layer_weights = np.random.normal(0, np.sqrt(2/neurons_previous), (neurons_current, neurons_previous))
	#		 # single_layer_bias = np.random.normal(0, np.sqrt(2/neurons_previous), (neurons_current, 1))


	def __init__(self):
		self._alfa = 0.09 			# learning rate
		self._layers_config = [784, 512, 10]
		self._layer_count = len(self._layers_config)
		self._weights = []
		self._bias = []
		self.train_path = './train/'

	def initial_weights(self):
		self._weights.append([])
		self._bias.append([])
		# average_weights.append([])
		# average_bias.append([])
		for i in range(1, self._layer_count ):
			pre_layer = self._layers_config[i-1]
			cur_layer = self._layers_config[i]
			layer_weights = np.random.normal(0, np.sqrt(2/pre_layer), (cur_layer, pre_layer))
			layer_bias = np.random.normal(0, np.sqrt(2/pre_layer), (cur_layer, 1))
			os.system('pause')
			self._weights.append(layer_weights)
			self._bias.append(layer_bias)
			# average_weights.append(layer_weights)
			# average_bias.append(layer_bias)

			print(layer_weights.shape)
			print(layer_bias.shape)

	def __save_weights(self):
		