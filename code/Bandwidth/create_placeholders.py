import tensorflow as tf
import numpy as np

def create_placeholders(n_x):

	"""
    Creates the placeholders for the tensorflow session.

    """


	X = tf.placeholder(tf.float32, [n_x, None])
	Y = tf.placeholder(tf.float32, [None]) #maybe should just be [None]?
	return X, Y