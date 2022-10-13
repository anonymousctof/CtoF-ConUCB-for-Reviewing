import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pandas as pd

sns.set(style="whitegrid")


import numpy as np
def average_10(folder_b_seed, alg, cluster_num, metric, interval, front):
    index_tmp = None
    val_list = []
    for seed in range(10):
        print("seed", seed)
        # file = f"{folder_b_seed}/{seed}/{alg}_evaluate-cluster_{cluster_num}.csv"
        file = f"{folder_b_seed}/{seed}/{alg}_evaluate-1008_interval_{interval}_acc-rs-mis/evaluate-cn_{cluster_num}.csv"
        df = pd.read_csv(file, header=0)
        index = df["iteration"].values
        index = index[:front]
        val = df[metric].values[:front]
        assert val.shape[0] == front, f"val{val}_front{front}"
        # print("index.shape[0]", index.shape[0])
        select = np.arange(0, index.shape[0], step=10)
        index = index[select]
        val = val[select]
        # print(index)
        if index_tmp is None:
            index_tmp = index
        else:
            if not np.array_equal(index_tmp, index):
                print("index_tmp", index_tmp)
                print("index", index)
                print(file)
                exit(1)
        val_list.append(val)
    val_arr = np.stack(val_list)
    # print(val_arr.shape)
    assert val_arr.shape[0] == 10
    val_mean = np.mean(val_arr, axis=0)
    val_mean = val_mean
    # assert val_arr.shape[1] == 50
    val_std_err = np.std(val_arr, axis=0) / np.sqrt(val_arr.shape[0])
    # print("val_mean.shape", val_mean.shape)
    return index_tmp, val_mean, val_std_err


# server 1

def x_update_scale_value(temp, position):
    result = temp // 1000
    return "{}k".format(int(result))

c6 = (214/255, 39/255, 39/255)
c5 = (237/255, 176/255, 30/255)

plot_sig = "1012_cluster_evaluation_summary"
folder_b_seed_alg1_alg2 = []
dataset = "MovieLens 25M"
folder_b_seed = "../../datasets/data3_movielens_generator/ml_25m_0401a1_un2000_in25000_0/1006_n1_final"

alg1 = "[Kmeans+del+5]+topk_[Kmeans_user_ts_cluster+Dyn+5.0+kalpha+0.1+ktalpha+0.25+lamb0.3+tlamb0.5+user_ts_a0.001+is_split_over_user_num].gz"
alg2 = "[Kmeans+del+5]++T_ratio0.01topk_[SKmeans+Dyn+10.0+salpha+0.25+salpha_p+1.4142135623730951+kalpha+0.1+ktalpha+0.25+lamb0.3+tlamb0.5+T_ratio0.01+is_split_over_user_num].gz"
folder_b_seed_alg1_alg2.append([folder_b_seed, alg1, alg2, dataset])

dataset = "LastFM"
folder_b_seed = "../../datasets/data4_lastfm/un_1500-bn_10000/train_item_user_test_rate/1006_n1_final"
alg1 = "[Kmeans+del+0.1]+topk_[Kmeans_user_ts_cluster+Dyn+1.0+kalpha+0.01+ktalpha+0.25+lamb0.5+tlamb0.1+user_ts_a0.01+is_split_over_user_num].gz"
alg2 = "[Kmeans+del+0.1]++T_ratio0.1topk_[SKmeans+Dyn+1.0+salpha+0.25+salpha_p+1.4142135623730951+kalpha+0.01+ktalpha+0.25+lamb0.5+tlamb0.1+T_ratio0.1+is_split_over_user_num].gz"
folder_b_seed_alg1_alg2.append([folder_b_seed, alg1, alg2, dataset])

fig = plt.figure(figsize=(8, 7.6))

title_id = ["a", "b", "c", "d"]
ax_list = fig.subplots(nrows=2, ncols=2)

def plot_one(folder_b_seed, alg1, alg2, dataset, loc):

    color1 = c6 # todo
    color2 = c5
    label1 = "CtoF-ConUCB+TS"
    label2 = "CtoF-ConUCB+"
    assert "Kmeans_user_ts" in alg1
    assert "SKmeans" in alg2

    interval = 50
    cluster_num_list = [5]
    metric_list = ["adjusted_rand_score", "adjusted_mutual_info_score"]
    # metric_list = ["adjusted_rand_score"]
    y_label_list = ["Adjusted Rand Index", "Adjusted Mutual Information"]
    y_label_sub = ["ARI", "AMI"]
    for j, metric in enumerate(metric_list):
        print("loc + j + 1", loc*2+j+1)
        ax = plt.subplot(2, 2, loc*2 + j + 1)
        l_list = []
        for i, cluster_num in enumerate(cluster_num_list):

            print("cluster_num", cluster_num)
            index1, mean1, ste1 = average_10(folder_b_seed, alg1, cluster_num, metric, interval, front)
            index2, mean2, ste2 = average_10(folder_b_seed, alg2, cluster_num, metric, interval, front)
            print(index1.shape)
            print(index2.shape)
            print(mean1.shape)
            print(mean2.shape)
            select = np.arange(0, len(index1), step=5)

            # ax = plt.gca()
            ax.fill_between(index2, mean2 - ste2,
                            mean2 + ste2, facecolors=color2, alpha=0.3)
                            # mean2 + ste2, facecolors='antiquewhite', alpha=0.5)
            ax.fill_between(index1, mean1 - ste1,
                            mean1 + ste1, facecolors=color1, alpha=0.3)
            # mean1 + ste1, facecolors='orangered', alpha=0.5)

            tmp = plt.plot(index2, mean2, label=label2, c=color2)
            l_list.append(tmp[0])
            tmp = plt.plot(index1, mean1, label=label1, c=color1)
            l_list.append(tmp[0])

            print(y_label_list[j])
            plt.ylabel(y_label_list[j])
            plt.xlabel("Time t")
            plt.ticklabel_format(style='sci', axis='both')
            plt.title(f"({title_id[loc * 2 + j]}) {y_label_sub[j]} of {dataset}")
            from matplotlib.ticker import FuncFormatter

            plt.gca().xaxis.set_major_formatter(FuncFormatter(x_update_scale_value))

        # ax.ticklabel_format(style='sci')
    return l_list[:2]


for loc, (folder_b_seed, alg1, alg2, title) in enumerate(folder_b_seed_alg1_alg2):
    l_list = plot_one(folder_b_seed, alg1[:-3], alg2[:-3], title, loc)
# fig.tight_layout()
line_labels = ['CtoF-ConUCB-Clu', 'CtoF-ConUCB-Clu+']
fig.legend(l_list,     # The line objects
           labels=line_labels,   # The labels for each line
           loc="upper center",   # Position of legend
           # borderaxespad=0.1,    # Small spacing around legend box
           # title="Legend Title",  # Title for the legend
           ncol=6,
            prop={'size': 8}
           )
plt.subplots_adjust(wspace =0.4, hspace =0.3, top=0.92, bottom=0.08, left=0.13, right=0.95)

plt.savefig(f"{plot_sig}.png")
plt.savefig(f"{plot_sig}.pdf")

plt.show()
