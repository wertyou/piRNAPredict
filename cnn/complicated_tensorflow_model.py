import tensorflow as tf


def deepnn(x):
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are gray scale.
    # It would be 3 for an RGB image, 4 for RGBA, etc.
    # TODO(Lin): try at least three combinations of arguments
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 1, 35, 5])

    # First convolutional layer - maps one grayscale image to 32 feature maps
    # The first two dimensions are the patch size, the next is the number of input channels and the last is the number of output channels.
    # TODO(Lin): Modify with reshape()
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([1, 8, 5, 16])
        b_conv1 = bias_variable([16])

        # 计算前向传播结果，并使用ReLU去线性化
        # Y_conv1, update_conv1 = batchnorm(conv2d(x_image, W_conv1), tst, iter, b_conv1, convolutional=True)
        # h_conv1 = tf.nn.relu(Y_conv1)
        # h_conv1 = tf.nn.dropout(h_conv1r, pkeep_conv, compatible_convolutional_noise_shape(h_conv1r))
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_1x2(h_conv1)

    # Second convolutional layer - maps 32 feature maps to 64.
    # TODO(Lin): Modify with conv1
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([1, 8, 16, 32])
        b_conv2 = bias_variable([32])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_1x2(h_conv2)

    # Third convolutional layer
    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([1, 8, 32, 64])
        b_conv3 = bias_variable([64])

        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

    # Third pooling layer
    with tf.name_scope('pool3'):
        h_pool3 = max_pool_1x2(h_conv3)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image is down to 7x7x64 maps -- maps this to 1024 features
    # TODO(Lin): Change this (why 1152? the change of dimension must be handled)
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([64 * 5, 256])
        b_fc1 = bias_variable([256])

        h_pool3_flat = tf.reshape(h_pool3, [-1, 64 * 5])

        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of features
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([256, 2])
        b_fc2 = bias_variable([2])

        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    return y_conv, keep_prob


def conv2d(x, W):
    """ conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """ max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def max_pool_1x2(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')


def weight_variable(shape):
    """ weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """ bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
