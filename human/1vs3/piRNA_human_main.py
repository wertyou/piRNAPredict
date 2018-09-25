from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import piRNA_datasets as input_data
import cross_validation as cv
import piRNA_human_model as model
import tensorflow as tf
import pandas as pd

# Human data
TRAIN_IMAGES = '..\..\source\Human_sequence3.txt'
TRAIN_LABELS = '..\..\source\Human_label3.txt'
DATA_NUM = 28800

FLAGS = None

# Parameters
LOGS_DIRECTORY = "logs/1vs3"
TOTAL_BATCH = 30000
learn_rate = 0.001
batch_size = 256
seeds_num = 20
test_size = 0.1

is_display = True
acc_list = []
auc_list = []
spec_list = []
recall_list = []
test_acc_list = []


def main(_):
    with tf.device('/gpu:0'):
        # Input
        x = tf.placeholder(tf.float32, [None, model.time_step, model.num_input])
        y_ = tf.placeholder(tf.float32, [None, model.num_class])

        # Create lstm model
        y_lstm, keep_prob = model.lstm(x)

        # Define loss
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_lstm)
        cross_entropy = tf.reduce_mean(cross_entropy)

        # Define optimizer
        with tf.name_scope('adam_optimizer'):
            train_step = tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy)

        # Create the node to calculate ccc
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y_lstm, 1), tf.argmax(y_, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

        # Create the node to calculate auc
        with tf.name_scope('auc'):
            labels = tf.reshape(tf.slice(tf.cast(y_, dtype=tf.bool), [0, 0], [-1, 1]), [-1])
            predictions = tf.reshape(
                tf.subtract(tf.slice(y_lstm, [0, 0], [-1, 1]), tf.slice(y_lstm, [0, 1], [-1, 1])),
                [-1])

            # Min Max Normalization
            Y_pred = (predictions - tf.reduce_min(predictions)) / (
                    tf.reduce_max(predictions) - tf.reduce_min(predictions))
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

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            for seed in range(1, seeds_num + 1):
                print('*' * 30, 'seed=', seed, '*' * 30)
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                summary_writer = tf.summary.FileWriter(LOGS_DIRECTORY, graph=tf.get_default_graph())

                sum_acc = 0
                sum_auc = 0
                sum_spec = 0
                sum_recall = 0
                record_fn = 0
                record_fp = 0
                training_accuracy_list = []

                all_piRNA = input_data.read_all(TRAIN_IMAGES, TRAIN_LABELS,
                                                test_size=test_size, seed=seed, is_display=is_display)
                test_accuracy_list = []
                for fold in range(10):

                    print('fold %d:' % fold)
                    piRNA = input_data.read_CV_datasets(fold, int(DATA_NUM * (1 - test_size)), all_piRNA)

                    for i in range(TOTAL_BATCH):
                        batch_x, batch_y = piRNA.train.next_batch(batch_size)
                        batch_x = batch_x.reshape(batch_size, model.time_step, model.num_input)

                        step, training_accuracy = sess.run([train_step, accuracy],
                                                           feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})

                        # print out results
                        if i % 50 == 0:
                            print('step %d, training accuracy %g' % (i, training_accuracy))
                            training_accuracy_list.append(training_accuracy)
                        if i % 1000 == 0:
                            print('test accuracy %g' % accuracy.eval(
                                feed_dict={x: piRNA.test.images.reshape(-1, model.time_step, model.num_input),
                                           y_: piRNA.test.labels, keep_prob: 1.0}))

                    auc, acc, recall, pred_neg, false_nega, false_posi, pred_pos = sess.run(
                        [roc_auc_update_op, acc_op, rec_op, all_neg, fn_op, fp_op, all_pos],
                        feed_dict={x: piRNA.validation.images.reshape(-1, model.time_step, model.num_input),
                                   y_: piRNA.validation.labels, keep_prob: 1.0})

                    # update specificity
                    current_fn = false_nega - record_fn
                    current_fp = false_posi - record_fp
                    true_nega = pred_neg - current_fn  # fp_op accumulate every loop
                    spec = true_nega / (true_nega + current_fp)
                    record_fn = false_nega
                    record_fp = false_posi

                    test_accuracy = accuracy.eval(
                        feed_dict={x: piRNA.test.images.reshape(-1, model.time_step, model.num_input),
                                   y_: piRNA.test.labels, keep_prob: 1.0})
                    test_accuracy_list.append(test_accuracy)

                    # Test Set
                    print('Test set accuracy %g' % test_accuracy)

                    # 10-CV metrices (acc, auc)
                    sum_acc = cv.acc(sum_acc, acc, fold, is_display=is_display)
                    sum_auc = cv.auc(sum_auc, auc, fold, is_display=is_display)
                    sum_spec = cv.spec(sum_spec, spec, fold, is_display=is_display)
                    sum_recall = cv.recall(sum_recall, recall, fold, is_display=is_display)
                test_accuracy_average = cv.average(test_accuracy_list)
                auc_average = cv.average(cv.auc_list)
                acc_average = cv.average(cv.acc_list)
                spec_average = cv.average(cv.spec_list)
                recall_average = cv.average(cv.recall_list)
                acc_list.append(acc_average)
                auc_list.append(auc_average)
                spec_list.append(spec_average)
                recall_list.append(recall_average)
                test_acc_list.append(test_accuracy_average)
                if is_display:
                    print('*** Test accuracy is:', test_accuracy_list)
                    print('*** The average test accuracy is:%.3f' % test_accuracy_average)
                    print('acc', acc_average)
                    print('auc', auc_average)
                    print('spec', spec_average)
                    print('recall', recall_average)
    data_frame = pd.DataFrame(
        {'AUC': auc_list, 'ACC': acc_list, 'SP': spec_list, 'SN': recall_list, 'Test ACC': test_acc_list})
    data_frame.to_csv('human1vs3.csv', index=True, columns=['AUC', 'ACC', 'SP', 'SN', 'Test ACC'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
