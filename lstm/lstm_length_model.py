import tensorflow as tf
import numpy as np

file_address = 'E:\pycharmWorkspace\piRNA\CollectiveIntelligence\program\\task1\\test\DrosophilaSequence.txt'
f = open(file_address, 'r')

a = f.read()

b = a.replace('\n', '')
dataset = eval(b)

# dataset = [[[0], [0], [0], [0], [1], [0]],
#            [[1], [0], [0]],
#            [[1], [0], [0], [0], [0]],
#            [[0], [1], [0], [0], [1]]
#            ]


feature_dim = 1

num_samples = len(dataset)
lengths = [len(s) for s in dataset]

print(lengths)

max_len = max(lengths)
print(max_len)

# 生成一个全零array来存放padding后的数据集
padding_dataset = np.zeros([num_samples, max_len, feature_dim])

# 将序列放入array中（相当于padding成一样长度）
for idx, seq in enumerate(dataset):
    padding_dataset[idx, :len(seq), :] = seq


# print(padding_dataset)

# cell = tf.contrib.rnn.BasicLSTMCell(num_units=64, state_is_tuple=True)
cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=2, state_is_tuple=True)


# sequence_length : 每个序列的长度    inputs :padding后的数据集
outputs, last_states = tf.nn.dynamic_rnn(
    cell=cell,
    dtype=tf.float64,
    sequence_length=lengths,
    inputs=padding_dataset)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    o = sess.run(outputs)
    s = sess.run(last_states)
    # print('output\n', o)

    # 从output中取最后一次输出
    print('last_o\n', o[:, -1, :])


def lstm_length():
    pass