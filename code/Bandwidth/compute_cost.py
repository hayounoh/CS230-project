import tensorflow as tf
import numpy as np

def compute_cost(Z_L, Y):
    cost = tf.reduce_mean(tf.square(Z_L-Y)) #L2 distance between guesses
    return cost