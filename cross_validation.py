acc_list = []
auc_list = []
spec_list = []
recall_list = []

# Get average data
def average(list):
    avg = 0.0
    n = len(list)
    for num in list:
        avg += num / n
    return avg


def acc(sum_acc, this_fold_accuracy, fold, is_display):
    print('---------------------------------------------')
    sum_acc += this_fold_accuracy
    print('10-CV Average Validation Set Accuracy %g' % (sum_acc / (fold + 1)))
    print('---------------------------------------------')
    acc_list.append(sum_acc / (fold + 1))
    if len(acc_list) == 10:
        if is_display:
            # display all accuracy data in a list
            print('****Accuracy list is :', acc_list)
            print('****The average accuracy is: %.3f' % average(acc_list))
    return sum_acc


def auc(sum_auc, this_fold_auc, fold, is_display):
    print('---------------------------------------------')
    sum_auc += this_fold_auc
    print('10-CV Average Validation set AUC %g' % (sum_auc / (fold + 1)))
    print('---------------------------------------------')
    auc_list.append(sum_auc / (fold + 1))
    if len(auc_list) == 10:
        if is_display:
            # display all accuracy data in a list
            print('****Auc list is :', auc_list)
            print('****The average auc is: %.3f' % average(auc_list))

    return sum_auc


def spec(sum_spec, this_fold_spec, fold, is_display):
    print('---------------------------------------------')
    sum_spec += this_fold_spec
    print('10-CV Average Validation set specificity %g' % (sum_spec / (fold + 1)))
    print('---------------------------------------------')
    spec_list.append(sum_spec / (fold + 1))
    if len(spec_list) == 10:
        if is_display:
            # display all accuracy data in a list
            print('****Specificity list is :', spec_list)
            print('****The average specificity is: %.3f' % average(spec_list))
    return sum_spec


def recall(sum_recall, this_fold_recall, fold, is_display):
    print('---------------------------------------------')
    sum_recall += this_fold_recall
    print('10-CV Average Validation set recall/sensitive %g' % (sum_recall / (fold + 1)))
    print('---------------------------------------------')
    recall_list.append(sum_recall / (fold + 1))
    if len(recall_list) == 10:
        if is_display:
            # display all accuracy data in a list
            print('****Recall/Sensitive list is :', recall_list)
            print('****The average recall/sensitive is: %.3f' % average(recall_list))
            print('---------------------------------------------')
    return sum_recall
