# Some code was borrowed from https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/models/image/mnist/convolutional.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


# Create model of CNN with slim api
def CNN(inputs, is_training=True):
    batch_norm_params = {'is_training': is_training, 'decay': 0.999, 'updates_collections': None}
    # arg_scope的作用范围内，是定义了指定层的默认参数，若想特别指定某些层的参数
    # 可以重新赋值（相当于重写）
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        x = tf.reshape(inputs, [-1, 1, 35, 5])

        # For slim.conv2d, default argument values are like
        # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
        # padding='SAME', activation_fn=nn.relu,
        # weights_initializer = initializers.xavier_initializer(),
        # biases_initializer = init_ops.zeros_initializer,

        # Model 1 With batch normalization
        # net = slim.conv2d(x, 32, [1, 10], scope='conv1')
        # net = slim.conv2d(net, 16, [1, 3], scope='conv2')
        # net = slim.max_pool2d(net, [1, 2], scope='pool1')
        # net = slim.conv2d(net, 32, [1, 5], scope='conv3')
        # net = slim.conv2d(net, 64, [1, 5], scope='conv4')
        # net = slim.max_pool2d(net, [1, 2], scope='pool2')
        # net = slim.flatten(net, scope='flatten3')

        # Model 2 Without batch normalization
        net = slim.conv2d(x, 32, [1, 10], normalizer_params=None, scope='conv1')
        net = slim.conv2d(net, 16, [1, 3], normalizer_params=None, scope='conv2')
        net = slim.max_pool2d(net, [1, 2], scope='pool1')
        net = slim.conv2d(net, 32, [1, 5], normalizer_params=None, scope='conv3')
        net = slim.conv2d(net, 64, [1, 5], normalizer_params=None, scope='conv4')
        net = slim.max_pool2d(net, [1, 2], scope='pool2')
        # slim.flatten扁平化输入向量！
        net = slim.flatten(net, scope='flatten3')

        # For slim.fully_connected, default argument values are like
        # activation_fn = nn.relu,
        # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
        # weights_initializer = initializers.xavier_initializer(),
        # biases_initializer = init_ops.zeros_initializer,
        net = slim.fully_connected(net, 64, normalizer_fn=None, scope='fc3')
        net = slim.dropout(net, is_training=is_training, scope='dropout3')  # 0.5 by default
        outputs = slim.fully_connected(net, 2, activation_fn=None, normalizer_fn=None, scope='fco')

        # 占位符，手动输入！
        keep_prob = tf.placeholder(tf.float32)
    return outputs, keep_prob
