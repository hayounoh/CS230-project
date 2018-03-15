import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from generate_signals import *
from load_dataset import *
from create_placeholders import *
from initialize_parameters import *
from forward_propagation import *
from compute_cost import *
import matplotlib.pyplot as plt

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001, num_epochs = 500, minibatch_size = 32, print_cost = True):

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(0)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = 1												# n_y : output size. always 1.
    costs = []                                        # To keep track of the cost
    
    X, Y = create_placeholders(n_x) #y is always a single number

    parameters = initialize_parameters(n_x)

    Z_L = forward_propagation(X, parameters)

    cost = compute_cost(Z_L, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(num_epochs):
            epoch_cost = 0
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1

            for i in range(num_minibatches):
                minibatch_X = X_train[:,i*minibatch_size:(i+1)*minibatch_size]
                minibatch_Y = Y_train[i*minibatch_size:(i+1)*minibatch_size]

                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                epoch_cost += minibatch_cost/num_minibatches
            if print_cost == True and epoch % 100 == 0:
                print('Cost after epoch %i: %f' % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
        
        # plot the cost
        plt.semilogy(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        accuracy = tf.reduce_mean(tf.square(tf.cast(Z_L-Y, 'float')))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters