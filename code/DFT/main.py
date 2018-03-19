import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

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
num_epochs = 7500
minibatch_size = 250
print_cost = True

parameters = model(X_train, Y_train, X_test, Y_test, percent_training, learning_rate, num_epochs, minibatch_size, print_cost)


'''
Timing
'''

num_examples = X_test.shape[1]


t_naive_DFT_start = time.time()
DFT_X_REAL = np.real(sc.linalg.dft(100).dot(X_test).T)
t_naive_DFT = time.time() - t_naive_DFT_start


t_fft_start = time.time()
FFT_X_REAL = np.real(np.fft.fft(X_test))
t_fft = time.time() - t_fft_start


W1 = parameters['W1']
b1 = parameters['b1']
W2 = parameters['W2']
b2 = parameters['b2']
W3 = parameters['W3']
b3 = parameters['b3']
W4 = parameters['W4']
b4 = parameters['b4']
W5 = parameters['W5']
b5 = parameters['b5']
W6 = parameters['W6']
b6 = parameters['b6']

t_nn_start = time.time()

Z1 = np.dot(W1, X_test) + b1
Z2 = np.dot(W2, Z1) + b2
Z3 = np.dot(W3, Z2) + b3

t_nn_DFT = time.time() - t_nn_start

print('Time for naive computation = %s seconds.' % (t_naive_DFT/num_examples))
print('Time for FFT computation = %s seconds.' % (t_fft/num_examples))
print('Time for NN computation = %s seconds.' % (t_nn_DFT/num_examples))





