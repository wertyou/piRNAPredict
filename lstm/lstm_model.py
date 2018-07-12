from tensorflow.contrib import rnn
import tensorflow as tf

# LSTM layer的层数
layer_num = 1
# 每个隐含层的节点数
hidden_size = 210
# keep_prob
keep_prob = tf.placeholder(tf.float32)
# 每行175个元素
num_input = 210
# 时序持续长度为1行，即每一次预测，需要输入1行
time_step = 1
# 二分类
num_class = 2

# Define Weights
weights = {
    'out': tf.Variable(tf.random_normal([hidden_size, num_class]))
}

# Define Biases
biases = {
    'out': tf.Variable(tf.random_normal([num_class]))
}


def lstm(x, is_training=True):
    # 矩阵分解
    x = tf.unstack(x, time_step, 1)

    # 定义一层Lstm_cell
    lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0)

    # 连接一个Dropout层
    if is_training:
        rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)

    # 两层神经网络结构
    cell = rnn.MultiRNNCell([lstm_cell] * layer_num, state_is_tuple=True)
    # Get lstm cell output
    outputs, states = rnn.static_rnn(cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    last_output = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return last_output, keep_prob
