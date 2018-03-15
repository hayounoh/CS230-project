import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from generate_signals import *
from load_dataset import *
from create_placeholders import *
from initialize_parameters import *
from forward_propagation import *
from compute_cost import *
from model import *

'''
Load Data
X has dimensions (time indices, number of examples)
Y has dimensions (number of examples,)
'''
num_examples = 5000
max_bandwidth = 20
max_num_sinusoids = 10

SavingDataToFile = False

if SavingDataToFile:
	data = generate_signals(num_examples, max_bandwidth, max_num_sinusoids)
	data_filename = 'data_m_%s_BW_%s_maxSinusoids_%s' % (num_examples, max_bandwidth, max_num_sinusoids)
	np.save(data_filename, data)
else:
	data = np.load('data_m_5000_BW_20_maxSinusoids_10.npy').item()

(X_train, Y_train, X_test, Y_test, t) = load_dataset(data)

learning_rate = 0.0001
num_epochs = 3500
minibatch_size = 20
print_cost = True

parameters = model(X_train, Y_train, X_test, Y_test, learning_rate, num_epochs, minibatch_size, print_cost)