from sklearn import metrics
import argparse
import json
import numpy as np
import pandas as pd
import os
parser = argparse.ArgumentParser(description='')
parser.add_argument('--fn', default=9, type=str)
args = parser.parse_args()
interval = 50
n_cluster = 5
iteration_list = [x for x in range(49, 65535, interval)]
file_plot = f"0616_interval_{interval}_acc-rs-mis"

file_ori = args.fn
arm_file = "/".join(file_ori.split('/')[:-3]) + "/user_preference.txt"

def load_user_prefer(file_name):
    theta = []
    with open(file_name, 'r') as fr:
        for line in fr:
            j_s = json.loads(line)
            theta_u = j_s['preference_v']
            theta.append(theta_u)
        theta = np.array(theta)
        theta = theta.squeeze()
    return theta

feature_gt = load_user_prefer(arm_file)
seed = 1345
max_iter = 200
from sklearn.cluster import KMeans
from vote_accuracy import calculate_vote_accuracy
kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(feature_gt)
labels_true = kmeans.labels_
evaluate_folder = f"{file_ori}_evaluate-{file_plot}/"
os.makedirs(evaluate_folder, exist_ok=True)
evaluate_file = f"{file_ori}_evaluate-{file_plot}/evaluate-cn_{n_cluster}.csv"

with open(evaluate_file, "w") as f:
    f.write(f"iteration,vote_acc,adjusted_rand_score,adjusted_mutual_info_score\n")

for iteration in iteration_list:
    file_label = f"{file_ori}cluster_belong_to.csv"
    labels_pred_all = pd.read_csv(file_label, header=None, index_col=0)

    labels_pred = labels_pred_all.loc[iteration].values
    assert labels_pred.shape[0] == 500
    if len(labels_pred.shape) >1 and labels_pred.shape[0] > 1:
        input()
        labels_pred = labels_pred[0]

    vote_acc = calculate_vote_accuracy(labels_true, labels_pred)
    adjusted_rand_score = metrics.adjusted_rand_score(labels_true, labels_pred)
    adjusted_mutual_info_score = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    with open(evaluate_file, "a+") as f:

        f.write(f"{iteration},{vote_acc},{adjusted_rand_score},{adjusted_mutual_info_score}\n")
