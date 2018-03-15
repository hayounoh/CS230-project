import tensorflow as tf
import numpy as np

def load_dataset(data):
	'''
	Given data (as a dict), load the data into training data, testing data.
	Also output the time indices.
	'''
	X = data['X']
	Y = data['Y']
	t = data['t']

	indices = np.random.permutation(Y.shape[0])

	percent_training = .9
	num_training = int(percent_training * Y.shape[0])

	training_idx, test_idx = indices[:num_training], indices[num_training:]

	X_train = X[training_idx,:]
	Y_train = Y[training_idx]

	X_test = X[test_idx, :]
	Y_test = Y[test_idx]

	return (X_train.T, Y_train, X_test.T, Y_test, t)