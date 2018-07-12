from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import piRNA_datasets as input_data
import cross_validation as cv
import lstm_model
import tensorflow as tf

# Human datas
# 1:1
# HUMAN_TRAIN_IMAGES = '..\lstm\source\HumanDatas\\1vs1\Human_sequence.txt'
# HUMAN_TRAIN_LABELS = '..\lstm\source\HumanDatas\\1vs1\Human_label.txt'
# HUMAN_TEST_IMAGES = '..\lstm\source\HumanDatas\\1vs1\Human_sequence_test.txt'
# HUMAN_TEST_LABELS = '..\lstm\source\HumanDatas\\1vs1\Human_label_test.txt'
# HUMAN_NUM = 13810

# 1:2
# HUMAN_TRAIN_IMAGES = '..\lstm\source\HumanDatas\\1vs2\Human_sequence.txt'
# HUMAN_TRAIN_LABELS = '..\lstm\source\HumanDatas\\1vs2\Human_label.txt'
# HUMAN_TEST_IMAGES = '..\lstm\source\HumanDatas\\1vs2\Human_sequence_test.txt'
# HUMAN_TEST_LABELS = '..\lstm\source\HumanDatas\\1vs2\Human_label_test.txt'
# HUMAN_NUM = 20715

# 1:3
HUMAN_TRAIN_IMAGES = '..\lstm\source\HumanDatas\\1vs3\Human_sequence.txt'
HUMAN_TRAIN_LABELS = '..\lstm\source\HumanDatas\\1vs3\Human_label.txt'
HUMAN_TEST_IMAGES = '..\lstm\source\HumanDatas\\1vs3\Human_sequence_test.txt'
HUMAN_TEST_LABELS = '..\lstm\source\HumanDatas\\1vs3\Human_label_test.txt'
HUMAN_NUM = 26800

# Mouse datas
# 1:1
# MOUSE_TRAIN_IMAGES = '..\lstm\source\MouseDatas\\1vs1\Mouse_sequence.txt'
# MOUSE_TRAIN_LABELS = '..\lstm\source\MouseDatas\\1vs1\Mouse_label.txt'
# MOUSE_TEST_IMAGES = '..\lstm\source\MouseDatas\\1vs1\Mouse_sequence_test.txt'
# MOUSE_TEST_LABELS = '..\lstm\source\MouseDatas\\1vs1\Mouse_label_test.txt'
# MOUSE_NUM = 26996

# 1:2
# MOUSE_TRAIN_IMAGES = '..\lstm\source\MouseDatas\\1vs2\Mouse_sequence.txt'
# MOUSE_TRAIN_LABELS = '..\lstm\source\MouseDatas\\1vs2\Mouse_label.txt'
# MOUSE_TEST_IMAGES = '..\lstm\source\MouseDatas\\1vs2\Mouse_sequence_test.txt'
# MOUSE_TEST_LABELS = '..\lstm\source\MouseDatas\\1vs2\Mouse_label_test.txt'
# MOUSE_NUM = 40494

# 1:3
MOUSE_TRAIN_IMAGES = '..\lstm\source\MouseDatas\\1vs3\Mouse_sequence.txt'
MOUSE_TRAIN_LABELS = '..\lstm\source\MouseDatas\\1vs3\Mouse_label.txt'
MOUSE_TEST_IMAGES = '..\lstm\source\MouseDatas\\1vs3\Mouse_sequence_test.txt'
MOUSE_TEST_LABELS = '..\lstm\source\MouseDatas\\1vs3\Mouse_label_test.txt'
MOUSE_NUM = 52000

# Drosophila datas
# 1:1
# DROSOPHILA_TRAIN_IMAGES = '..\lstm\source\DrosophilaDatas\\1vs1\Drosophila_sequence.txt'
# DROSOPHILA_TRAIN_LABELS = '..\lstm\source\DrosophilaDatas\\1vs1\Drosophila_label.txt'
# DROSOPHILA_TEST_IMAGES = '..\lstm\source\DrosophilaDatas\\1vs1\Drosophila_sequence_test.txt'
# DROSOPHILA_TEST_LABELS = '..\lstm\source\DrosophilaDatas\\1vs1\Drosophila_label_test.txt'
# DROSOPHILA_NUM = 17428

# 1:2
# DROSOPHILA_TRAIN_IMAGES = '..\lstm\source\DrosophilaDatas\\1vs2\Drosophila_sequence.txt'
# DROSOPHILA_TRAIN_LABELS = '..\lstm\source\DrosophilaDatas\\1vs2\Drosophila_label.txt'
# DROSOPHILA_TEST_IMAGES = '..\lstm\source\DrosophilaDatas\\1vs2\Drosophila_sequence_test.txt'
# DROSOPHILA_TEST_LABELS = '..\lstm\source\DrosophilaDatas\\1vs2\Drosophila_label_test.txt'
# DROSOPHILA_NUM = 26142

# 1:3
DROSOPHILA_TRAIN_IMAGES = '..\lstm\source\DrosophilaDatas\\1vs3\Drosophila_sequence.txt'
DROSOPHILA_TRAIN_LABELS = '..\lstm\source\DrosophilaDatas\\1vs3\Drosophila_label.txt'
DROSOPHILA_TEST_IMAGES = '..\lstm\source\DrosophilaDatas\\1vs3\Drosophila_sequence_test.txt'
DROSOPHILA_TEST_LABELS = '..\lstm\source\DrosophilaDatas\\1vs3\Drosophila_label_test.txt'
DROSOPHILA_NUM = 28472


FLAGS = None

LOGS_DIRECTORY = "logs/train"
TOTAL_BATCH = 10000
# TOTAL_BATCH = 20000
# TOTAL_BATCH = 50000
batch_size = 128


def main(_):
    x = tf.placeholder(tf.float32, [None, lstm_model.time_step, lstm_model.num_input])
    y_ = tf.placeholder(tf.float32, [None, lstm_model.num_class])

    # Create the model
    y_lstm, keep_prob = lstm_model.lstm(x)

    # Define loss
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_lstm)
    cross_entropy = tf.reduce_mean(cross_entropy)

    # Define optimizer
    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # Create the node to calculate ccc
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_lstm, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    # Create the node to calculate auc
    with tf.name_scope('auc'):
        labels = tf.reshape(tf.slice(tf.cast(y_, dtype=tf.bool), [0, 0], [-1, 1]), [-1])
        predictions = tf.reshape(tf.subtract(tf.slice(y_lstm, [0, 0], [-1, 1]), tf.slice(y_lstm, [0, 1], [-1, 1])),
                                 [-1])

        # Min Max Normalization
        Y_pred = (predictions - tf.reduce_min(predictions)) / (tf.reduce_max(predictions) - tf.reduce_min(predictions))
        roc_auc, roc_auc_update_op = tf.metrics.auc(labels, Y_pred, curve='ROC', name='roc')

    # Create the node to calculate acc
    with tf.name_scope('metrics'):
        acc, acc_op = tf.metrics.accuracy(tf.argmax(y_, 1), tf.argmax(y_lstm, 1))
        rec, rec_op = tf.metrics.recall(tf.argmax(y_, 1), tf.argmax(y_lstm, 1))

        all_pos = tf.reduce_sum(tf.argmin(y_lstm, 1))
        all_neg = tf.reduce_sum(tf.argmax(y_lstm, 1))
        fn, fn_op = tf.metrics.false_negatives(tf.argmax(y_, 1), tf.argmax(y_lstm, 1))
        fp, fp_op = tf.metrics.false_positives(tf.argmax(y_, 1), tf.argmax(y_lstm, 1))

    # Add ops to save and restore all the variables
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        summary_writer = tf.summary.FileWriter(LOGS_DIRECTORY, graph=tf.get_default_graph())

        max_acc = 0
        sum_acc = 0
        sum_auc = 0
        sum_spec = 0
        sum_recall = 0
        record_fn = 0
        record_fp = 0
        # shuffle the training set

        all_piRNA = input_data.read_all(DROSOPHILA_NUM, DROSOPHILA_TRAIN_IMAGES, DROSOPHILA_TRAIN_LABELS,
                                        DROSOPHILA_TEST_IMAGES, DROSOPHILA_TEST_LABELS)
        # all_piRNA = input_data.read_all(MOUSE_NUM, MOUSE_TRAIN_IMAGES, MOUSE_TRAIN_LABELS,
        #                                 MOUSE_TEST_IMAGES, MOUSE_TEST_LABELS)
        # all_piRNA = input_data.read_all(HUMAN_NUM, HUMAN_TRAIN_IMAGES, HUMAN_TRAIN_LABELS,
        #                                 HUMAN_TEST_IMAGES, HUMAN_TEST_LABELS)

        for fold in range(5):

            print('fold %d:' % fold)
            piRNA = input_data.read_CV_datasets(fold, DROSOPHILA_NUM, all_piRNA)
            # piRNA = input_data.read_CV_datasets(fold, MOUSE_NUM, all_piRNA)
            # piRNA = input_data.read_CV_datasets(fold, HUMAN_NUM, all_piRNA)

            for i in range(TOTAL_BATCH):
                # 改动
                # batch = piRNA.train.next_batch(50)
                batch_x, batch_y = piRNA.train.next_batch(batch_size)
                batch_x = batch_x.reshape(batch_size, lstm_model.time_step, lstm_model.num_input)

                # step, training_accuracy = sess.run([train_step, accuracy],
                # feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
                step, training_accuracy = sess.run([train_step, accuracy],
                                                   feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})

                # print out results
                if i % 50 == 0:
                    print('step %d, training accuracy %g' % (i, training_accuracy))
                if i % 1000 == 0:
                    print('test accuracy %g' % accuracy.eval(
                        feed_dict={x: piRNA.test.images.reshape(-1, lstm_model.time_step, lstm_model.num_input),
                                   y_: piRNA.test.labels, keep_prob: 1.0}))

            auc, acc, recall, pred_neg, false_nega, false_posi, pred_pos = sess.run(
                [roc_auc_update_op, acc_op, rec_op, all_neg, fn_op, fp_op, all_pos],
                feed_dict={x: piRNA.validation.images.reshape(-1, lstm_model.time_step, lstm_model.num_input),
                           y_: piRNA.validation.labels, keep_prob: 1.0})

            # update specificity
            current_fn = false_nega - record_fn
            current_fp = false_posi - record_fp
            true_nega = pred_neg - current_fn  # fp_op accumulate every loop
            spec = true_nega / (true_nega + current_fp)
            record_fn = false_nega
            record_fp = false_posi

            # Test Set
            print('Test Set accuracy %g' % accuracy.eval(
                feed_dict={x: piRNA.test.images.reshape(-1, lstm_model.time_step, lstm_model.num_input),
                           y_: piRNA.test.labels, keep_prob: 1.0}))

            # 5-CV metrices (acc, auc)
            sum_acc = cv.acc(sum_acc, acc, fold)
            sum_auc = cv.auc(sum_auc, auc, fold)
            sum_spec = cv.spec(sum_spec, spec, fold)
            sum_recall = cv.recall(sum_recall, recall, fold)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
