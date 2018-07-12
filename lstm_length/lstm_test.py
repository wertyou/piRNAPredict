import numpy as np
import tensorflow as tf

max_length = 175
feature_dim = 1
time_step = 1
# keep_prob
keep_prob = tf.placeholder(tf.float32)

def lstm_length(x):
    # 矩阵分解
    x = tf.unstack(x, time_step, 1)
    x = str(x)
    x = x.replace('\n', '')
    # print(x)
    da = eval(x)
    # 生成一个全零array来存放padding后的数据集
    padding_dataset = np.zeros([1, max_length, feature_dim])
    # print(padding_dataset)

    # 将序列放入array中（相当于padding成一样长度）
    for idx, seq in enumerate(da):
        padding_dataset[idx, :len(seq), :] = seq

    # cell = tf.contrib.rnn.BasicLSTMCell(num_units=64, state_is_tuple=True)
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=2, state_is_tuple=True)

    # sequence_length : 每个序列的长度    inputs :padding后的数据集
    # ！！！！sequence_length : 这里读取数据，读取文件的所有行长度保存下来。
    # inputs 这里的BUG处理一下！！！
    outputs, last_states = tf.nn.dynamic_rnn(
        cell=cell,
        dtype=tf.float64,
        sequence_length=len(x),
        inputs=padding_dataset)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        o = sess.run(outputs)
        # s = sess.run(last_states)
        # print('output\n', o)

        # 从output中取最后一次输出
        last_out = o[:, -1, :]
        # print('last_o\n', o[:, -1, :])

    return last_out, keep_prob
