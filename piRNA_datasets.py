import numpy as np
import collections
from sklearn.model_selection import train_test_split



Datasets = collections.namedtuple('Dataset', ['train', 'validation', 'test'])


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


def read_all(SEQUENCE_FILE_ADDRESS, LABEL_FILE_ADDRESS, test_size, seed, is_display):
    """

    :param SEQUENCE_FILE_ADDRESS: the sequence file address
    :param LABEL_FILE_ADDRESS: the label file address
    :param test_size: the ration of test data/all data
    :param seed: seed
    :param is_display:  display the data shape
    :return: train_images, train_labels, test_images, test_labels
    """

    sequence = np.loadtxt(SEQUENCE_FILE_ADDRESS, delimiter=' ', dtype='float32')
    label = np.loadtxt(LABEL_FILE_ADDRESS, delimiter=' ', dtype='float32')

    # load sequence matrix
    train_images, test_images, train_labels, test_labels = train_test_split(
        sequence, label, test_size=test_size, random_state=seed
    )
    # np.savetxt('human_train.txt', train_images, fmt='%d')
    # np.savetxt('human_label.txt', train_labels, fmt='%d')

    if is_display:
        print('Train:  Train data shape is :', train_images.shape, ', Train label shape is:', test_images.shape)
        print('Test:   Test data shape is :', train_labels.shape, ', Test label shape is:', test_labels.shape)
        print('*' * 100)
    return train_images, train_labels, test_images, test_labels


def read_CV_datasets(fold, DATA_NUMBER, all_datasets):
    VALIDATION_SIZE = DATA_NUMBER // 10

    train_images = all_datasets[0]
    train_labels = all_datasets[1]
    test_images = all_datasets[2]
    test_labels = all_datasets[3]

    validation_images = train_images[fold * VALIDATION_SIZE: fold * VALIDATION_SIZE + VALIDATION_SIZE]
    validation_labels = train_labels[fold * VALIDATION_SIZE: fold * VALIDATION_SIZE + VALIDATION_SIZE]

    train_range = list(set(range(DATA_NUMBER)).difference(
        set(range(fold * VALIDATION_SIZE, fold * VALIDATION_SIZE + VALIDATION_SIZE))))
    train_images = train_images[train_range]
    train_labels = train_labels[train_range]

    train = DataSet(train_images, train_labels)
    validation = DataSet(validation_images, validation_labels)
    test = DataSet(test_images, test_labels)

    return Datasets(train=train, validation=validation, test=test)
