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
num_examples = 30000
max_bandwidth = 10
max_num_sinusoids = 10

SavingDataToFile = False

if SavingDataToFile:
	data = generate_signals(num_examples, max_bandwidth, max_num_sinusoids)
	data_filename = '/Users/jonathantuck/School/GRADUATE/CS 230/Data/DFT_data_m_%s_BW_%s_maxSinusoids_%s'  % (num_examples, max_bandwidth, max_num_sinusoids)
	np.save(data_filename, data)
else:
	data = np.load('/Users/jonathantuck/School/GRADUATE/CS 230/Data/DFT_data_m_30000_BW_10_maxSinusoids_10.npy').item()

(X_train, Y_train, X_test, Y_test, t, percent_training) = load_dataset(data)

learning_rate = 0.001
num_epochs = 2500
minibatch_size = 250
print_cost = True

parameters = model(X_train, Y_train, X_test, Y_test, percent_training, learning_rate, num_epochs, minibatch_size, print_cost)