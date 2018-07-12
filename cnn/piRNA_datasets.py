import numpy as np
import collections

# namedtuple("名称"，[属性list])
Datasets = collections.namedtuple('Dataset', ['train', 'validation', 'test'])

# @property 装饰器可以将返回变量设为只读，防止修改变量引发的问题！
class DataSet(object):

    def __init__(self,
                 images,
                 labels):
        assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def shuffle(self):
        perm0 = np.arange(self._num_examples)
        np.random.shuffle(perm0)
        self._images = self.images[perm0]
        self._labels = self.labels[perm0]

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        start = self._index_in_epoch

        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]

        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]


def read_all(TRAIN_NUMBER, TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS, reshape=True, seed=None):
    # Human
    # TRAIN_IMAGES = '../../source/SparseProfileFeatureHumanINT.txt'
    # TRAIN_LABELS = '../../source/labelHuman.txt'
    # TRAIN_IMAGES = '../../source/drosophila_sequence.txt'
    # TRAIN_LABELS = '../../source/drosophila_label.txt'
    # TEST_IMAGES = '../../source/human_test_sequences.txt'
    # TEST_LABELS = '../../source/human_test_labels.txt'

    # load sequence matrix
    train_images = np.loadtxt(TRAIN_IMAGES, delimiter=' ', dtype='float32')
    train_labels = np.loadtxt(TRAIN_LABELS, delimiter=' ', dtype='float32')

    test_images = np.loadtxt(TEST_IMAGES, delimiter=' ', dtype='float32')
    test_labels = np.loadtxt(TEST_LABELS, delimiter=' ', dtype='float32')

    perm0 = np.arange(TRAIN_NUMBER)
    # 随机打乱数据顺序
    np.random.shuffle(perm0)
    train_images = train_images[perm0]
    train_labels = train_labels[perm0]

    return train_images, train_labels, test_images, test_labels


def read_CV_datasets(fold, TRAIN_NUMBER, all_datasets):
    # dtype=dtype.float32,
    # reshape=True,
    # seed=None):
    # human
    # TRAIN_NUMBER = 13810

    # mouse
    # TRAIN_NUMBER = 18428
    # TODO 改动3
    TEST_NUMBER = 1000
    VALIDATION_SIZE = TRAIN_NUMBER // 5

    train_images = all_datasets[0]
    train_labels = all_datasets[1]
    test_images = all_datasets[2]
    test_labels = all_datasets[3]

    validation_images = train_images[fold * VALIDATION_SIZE: fold * VALIDATION_SIZE + VALIDATION_SIZE]
    validation_labels = train_labels[fold * VALIDATION_SIZE: fold * VALIDATION_SIZE + VALIDATION_SIZE]

    train_range = list(set(range(TRAIN_NUMBER)).difference(
        set(range(fold * VALIDATION_SIZE, fold * VALIDATION_SIZE + VALIDATION_SIZE))))
    train_images = train_images[train_range]
    train_labels = train_labels[train_range]

    train = DataSet(train_images, train_labels)
    validation = DataSet(validation_images, validation_labels)
    test = DataSet(test_images, test_labels)

    return Datasets(train=train, validation=validation, test=test)
