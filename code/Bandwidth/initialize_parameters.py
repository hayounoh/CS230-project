import tensorflow as tf
import numpy as np

def initialize_parameters(len_example):
	"""
	Initializes parameters to build a neural network with tensorflow.

	Inputs:
	len_example: the length of a single example.

	Returns:
	parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3, W4, b4
	"""

	tf.set_random_seed(0)
	    
	### START CODE HERE ### (approx. 6 lines of code)
	W1 = tf.get_variable("W1", [100,len_example], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	b1 = tf.get_variable("b1", [100,1], initializer = tf.zeros_initializer())
	W2 = tf.get_variable("W2", [50,100], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	b2 = tf.get_variable("b2", [50,1], initializer = tf.zeros_initializer())
	W3 = tf.get_variable("W3", [25,50], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	b3 = tf.get_variable("b3", [25,1], initializer = tf.zeros_initializer())
	W4 = tf.get_variable("W4", [12,25], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	b4 = tf.get_variable("b4", [12,1], initializer = tf.zeros_initializer())
	W5 = tf.get_variable("W5", [6,12], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	b5 = tf.get_variable("b5", [6,1], initializer = tf.zeros_initializer())
	W6 = tf.get_variable("W6", [1,6], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	b6 = tf.get_variable("b6", [1,1], initializer = tf.zeros_initializer())
	### END CODE HERE ###

	parameters = {"W1": W1,
	              "b1": b1,
	              "W2": W2,
	              "b2": b2,
	              "W3": W3,
	              "b3": b3,
	              "W4": W4,
	              "b4": b4,
	              "W5": W5,
	              "b5": b5,
	              "W6": W6,
	              "b6": b6}
	return parameters