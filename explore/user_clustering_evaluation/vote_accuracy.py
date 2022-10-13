
import pandas as pd
import numpy as np

def calculate_vote_accuracy(gt, cur):
    assert gt.shape == cur.shape
    assert isinstance(gt, np.ndarray)
    assert isinstance(cur, np.ndarray)
    accurate_num = 0
    wrong_num = 0
    cur_set = set(cur)
    for cur_label in cur_set:
        index = cur == cur_label
        gt_this_cluster = gt[index]
        label_freq_most = np.bincount(gt_this_cluster).argmax()
        accurate_num_tmp = np.sum(label_freq_most == gt_this_cluster)
        wrong_num_tmp = np.sum(label_freq_most != gt_this_cluster)
        assert accurate_num_tmp + wrong_num_tmp == gt_this_cluster.shape[0]
        accurate_num += accurate_num_tmp
        wrong_num += wrong_num_tmp
    assert accurate_num + wrong_num == gt.shape[0]
    acc = accurate_num / gt.shape[0]
    return acc


if __name__ == "__main__":
    gt = np.array([5, 5, 5, 4, 4, 4])
    cur = np.array([3, 3, 2, 2, 1, 1])
    acc = calculate_vote_accuracy(gt, cur)
    print("acc", acc)