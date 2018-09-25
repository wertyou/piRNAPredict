from tensorflow.contrib import rnn
import tensorflow as tf
import numpy as np

# LSTM layer
layer_num = 3
# Hidden size
hidden_size = 500
# Keep_prob
keep_prob = tf.placeholder(tf.float32)
# Input number
num_input = 175
# Time step
time_step = 1
# 2 class
num_class = 2
# Learning rate
learning_rate = 0.001
# Batch_size
batch_size = 128

# Xaiver init
def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


# Define Weights
weights = {
    'out': tf.Variable(tf.random_normal([hidden_size, num_class]))
    # 'out': tf.Variable(xavier_init(hidden_size, num_class))
}

# Define Biases
biases = {
    'out': tf.Variable(tf.random_normal([num_class]))
}


def lstm_cell(hidden_size, keep_prob):
    cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0)
    return rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)


def lstm(x):

    x = tf.unstack(x, time_step, 1)

    # The multi lstm layer
    lstm_cells = rnn.MultiRNNCell([lstm_cell(hidden_size, keep_prob) for _ in range(layer_num)], state_is_tuple=True)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cells, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    last = tf.matmul(outputs[-1], weights['out']) + biases['out']

    last_output = tf.nn.dropout(last, keep_prob)
    return last_output, keep_prob
