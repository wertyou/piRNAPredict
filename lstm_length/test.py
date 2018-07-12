import numpy as np

# dataset = [
#     [[0], [0], [1], [0], [0], [1], [0], [0], [0], [0], [1], [0], [0], [0], [0], [0], [1], [0], [0], [0], [0], [0], [0],
#      [0], [1], [0], [1], [0], [0], [0], [0], [0], [1], [0], [0], [0], [0]
#         , [0], [0], [1], [0], [0], [0], [0], [1], [0], [0], [1], [0], [0], [0], [1], [0], [0], [0], [0], [1], [0], [0],
#      [0], [0], [0], [0], [0], [1], [0], [0], [0], [0], [1], [0], [0], [0], [0]
#         , [1], [0], [0], [1], [0], [0], [1], [0], [0], [0], [0], [0], [0], [0], [0], [1], [0], [0], [0], [0], [1], [0],
#      [1], [0], [0], [0]], [[0], [1]]]
f = open(r"E:\pycharmWorkspace\piRNA\CollectiveIntelligence\program\task1\lstm_length\Drosophila_sequence_test.txt")
data = f.read()
dataset = eval(data)


feature_dim = 1

num_samples = len(dataset)
print(num_samples)
lengths = [len(s) for s in dataset]

print(lengths)

max_len = max(lengths)
print(max_len)

# 生成一个全零array来存放padding后的数据集
padding_dataset = np.zeros([num_samples, max_len, feature_dim])

# 将序列放入array中（相当于padding成一样长度）
# for idx, seq in enumerate(dataset):
#     padding_dataset[idx, :len(seq), :] = seq

print(padding_dataset)
