# -*- coding: utf-8 -*-
# author: Weiran Lin

import numpy as np
from numpy import array
import time
import sys

# parameters
max_length = 33


def getRawSequences(f):
    seqslst = []
    while True:
        s = f.readline()
        if not s:
            break
        else:
            if '>' not in s:
                seq = s.split('\n')[0]
                seqslst.append(seq)
    return seqslst


# 'A': array([1, 0, 0, 0, 0]),
# 'C': array([0, 1, 0, 0, 0]), 
# 'G': array([0, 0, 1, 0, 0]),
# 'T': array([0, 0, 0, 1, 0]), 
# 'E': array([0, 0, 0, 0, 1]), 

############################### Sparse Profile ################################
def getSparseProfile(instances, alphabet, vdim):
    sparse_dict = getSparseDict(alphabet)
    for i in range(len(alphabet)):
        print(sparse_dict[alphabet[i]])
    X = []
    for sequence in instances:
        vector = getSparseProfileVector(sequence, sparse_dict, vdim)
        X.append(vector)
    X = array(X, dtype=object)
    return X


def getSparseDict(alphabet):  # get one-hot vector
    alphabet_num = len(alphabet)
    identity_matrix = np.eye(alphabet_num + 1, dtype=int)
    sparse_dict = {alphabet[i]: identity_matrix[i] for i in range(alphabet_num)}
    sparse_dict['E'] = identity_matrix[alphabet_num]
    return sparse_dict


def getSparseProfileVector(sequence, sparse_dict, vdim):
    seq_length = len(sequence)
    sequence = sequence + 'E' * (vdim - seq_length) if seq_length <= vdim else sequence[0:vdim]
    vector = sparse_dict.get(sequence[0])
    for i in range(1, vdim):
        temp = sparse_dict.get(sequence[i])
        vector = np.hstack((vector, temp))
    return vector


if __name__ == '__main__':
    # global posi_samples_file
    # global nega_samples_file
    posi_samples_file = sys.argv[1]
    nega_samples_file = sys.argv[2]
    animal = posi_samples_file.split('_')[0]
    fp = open(posi_samples_file, 'r')
    posis = getRawSequences(fp)  # read the raw sequence
    fn = open(nega_samples_file, 'r')
    negas = getRawSequences(fn)
    instances = array(posis + negas)
    alphabet = ['A', 'C', 'G', 'T']

    # label
    Y = array([1] * len(posis) + [0] * len(negas), dtype=int)
    np.savetxt(animal + '_label.txt', Y, fmt='%d')

    # Sparse Profile
    print('......................')
    print('coding for feature: Sparse Profile, beginning')
    tic = time.clock()
    X = getSparseProfile(instances, alphabet, vdim=max_length)
    np.savetxt(animal + '_sequence.txt', X, fmt='%s')  # fmt='%d' specofies that output int
    toc = time.clock()
    print('Coding time:%.3f minutes' % ((toc - tic) / 60))
