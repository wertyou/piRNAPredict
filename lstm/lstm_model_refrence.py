from __future__ import print_function

from tensorflow.contrib import rnn
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# import MNIST data
mnist = input_data.read_data_sets("tmp/data", one_hot=True)

# Training Parameters
learning_rate = 0.001
training_steps = 5000
batch_size = 128
display_step = 200

# Network Parameters
num_input = 28  # img shape 28*28
timesteps = 28
num_hidden = 128
num_class = 10

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_class])

# Define Weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_class]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_class]))
}


# Define RNN
def RNN(x, weights, biases):
    # 矩阵分解
    x = tf.unstack(x, timesteps, 1)
    print(x)
    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimier
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y
))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

train_op = optimizer.minimize(loss_op)

# Evaluate model()
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializer the variables
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps + 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape(batch_size, timesteps, num_input)

        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calcuate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={
                X: batch_x, Y: batch_y
            })
            print("Step" + str(step) + ",Minibatch loss=" +
                  "{:.4f}".format(loss) + ", Training Accuracy= " +
                  "{:.3f}".format(acc)
                  )
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
