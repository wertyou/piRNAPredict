def acc(sum_acc, this_fold_accuracy, fold):
    print('---------------------------------------------')
    sum_acc += this_fold_accuracy
    print('5-CV Average Validation Set Accuracy %g' % (sum_acc / (fold + 1)))
    print('---------------------------------------------')
    return sum_acc


def auc(sum_auc, this_fold_auc, fold):
    print('---------------------------------------------')
    sum_auc += this_fold_auc
    print('5-CV Average Validation set AUC %g' % (sum_auc / (fold + 1)))
    print('---------------------------------------------')
    return sum_auc


def spec(sum_spec, this_fold_spec, fold):
    print('---------------------------------------------')
    sum_spec += this_fold_spec
    print('5-CV Average Validation set specificity %g' % (sum_spec / (fold + 1)))
    print('---------------------------------------------')
    return sum_spec


def recall(sum_recall, this_fold_recall, fold):
    print('---------------------------------------------')
    sum_recall += this_fold_recall
    print('5-CV Average Validation set recall %g' % (sum_recall / (fold + 1)))
    print('---------------------------------------------')
    return sum_recall
