
import numpy as np
import pandas as pd
import os
import gzip
import matplotlib.pyplot as plt
import seaborn as sns
def cal_improve_from_index_last(L):
    improve_list = []
    for i in range(1, len(L)):
        # print(i)
        # improve = (L[i] - L[i-1]) / L[i - 1]
        improve = (L[i] - L[0]) / L[0]
        improve_list.append(improve)
    return improve_list

data = dict()
front = 2**16 - 1
# sig = "0601_real_missing_key_n"
sig = "1012_missing_key_summary"



c1 = (31/255, 119/255, 180/255)
c2 = (139/255, 139/255, 139/255)
c3 = (38/255, 217/255, 175/255)
c4 = (125/255, 46/255, 140/255)
c5 = (237/255, 176/255, 30/255)
c6 = (214/255, 39/255, 39/255)
folder_datasetname_conf = []
folder_path = "../../dataset-generate/data4_movielens_generator/ml_25m_0401a1_un2000_in25000_0/1006_n1_final"

dataset_name = "(a) MovieLens"
algs_plot_conf =[
                ('[Item+1.0]+topk_[ConUCB+alpha+0.01+lamb0.5+talpha0.25+tlamb0.9]-selection.gz', 'ConUCB', c1),
                ('[Item+0.4]+del_k_high+topk_[ConUCB+alpha+0.01+lamb0.5+talpha0.25+tlamb0.9]-selection.gz', 'ConUCB (40%)', c2),
                ('[Item+0.6]+del_k_high+topk_[ConUCB+alpha+0.01+lamb0.5+talpha0.25+tlamb0.9]-selection.gz', 'ConUCB (60%)', c3),
                ('[Kmeans+del+5]+topk_[ConUCB+alpha+0.1+lamb0.3+talpha0.25+tlamb0.5]-selection.gz', 'CtoF-ConUCB', c4),
                ('[Kmeans+del+5]++T_ratio0.01topk_[SKmeans+Dyn+10.0+salpha+0.25+salpha_p+1.4142135623730951+kalpha+0.1+ktalpha+0.25+lamb0.3+tlamb0.5+T_ratio0.01+is_split_over_user_num].gz', 'CtoF-ConUCB-Clu', c5),
                ('[Kmeans+del+5]+topk_[Kmeans_user_ts_cluster+Dyn+5.0+kalpha+0.1+ktalpha+0.25+lamb0.3+tlamb0.5+user_ts_a0.001+is_split_over_user_num].gz', 'CtoF-ConUCB-Clu+', c6),
                ]
folder_datasetname_conf.append((folder_path, dataset_name, algs_plot_conf))

folder_path = "./../dataset-generate/data5_lastfm/un_1500-bn_10000/train_item_user_test_rate/1006_n1_final"

dataset_name = "(b) LastFM"
algs_plot_conf =[
                ('[Item+1.0]+topk_[ConUCB+alpha+1.0+lamb0.5+talpha0.25+tlamb0.1]-selection.gz', 'ConUCB', c1),
                ('[Item+0.4]+del_k_high+topk_[ConUCB+alpha+1.0+lamb0.5+talpha0.25+tlamb0.1]-selection.gz', 'ConUCB (40%)', c2),
                ('[Item+0.6]+del_k_high+topk_[ConUCB+alpha+1.0+lamb0.5+talpha0.25+tlamb0.1]-selection.gz', 'ConUCB (60%)', c3),
                ('[Kmeans+del+0.1]+topk_[ConUCB+alpha+0.01+lamb0.9+talpha0.25+tlamb0.1]-selection.gz', 'CtoF-ConUCB', c4),
                ('[Kmeans+del+0.1]++T_ratio0.1topk_[SKmeans+Dyn+1.0+salpha+0.25+salpha_p+1.4142135623730951+kalpha+0.01+ktalpha+0.25+lamb0.5+tlamb0.1+T_ratio0.1+is_split_over_user_num].gz', 'CtoF-ConUCB-Clu', c5),
                ('[Kmeans+del+0.1]+topk_[Kmeans_user_ts_cluster+Dyn+1.0+kalpha+0.01+ktalpha+0.25+lamb0.5+tlamb0.1+user_ts_a0.01+is_split_over_user_num].gz', 'CtoF-ConUCB-Clu+', c6),
                ]
folder_datasetname_conf.append((folder_path, dataset_name, algs_plot_conf))

print(folder_datasetname_conf)

dataset_name_to_save = "-".join(folder_path.strip().split('/')[-2:-1])


def x_update_scale_value(temp, position):
    result = temp // 1000
    return "{}k".format(int(result))

x = np.array([1, 2, 3, 4])
y = np.array([1, 4, 9, 16])
# fig, ax_list = plt.subplots(1, 4, figsize=(15,3.5))
fig = plt.figure(figsize=(7.5, 3.5))
ax_list = fig.subplots(nrows=1, ncols=2)

sns.set(style="whitegrid")
def draw(folder_path, dataset_name, algs_plot_conf, location_subplot):
    print("00000000000")
    print(algs_plot_conf)
    ax = plt.subplot(1, 2, location_subplot+1)
    l_list = []

    algs_data = {}
    for seed in range(10):
        # print(seed)
        for file_name, algs_name, color in algs_plot_conf:
            # print("algs_name", algs_name)
            file_path = f"{folder_path}/{seed}/{file_name}"
            try:
                df = pd.read_csv(gzip.open(file_path, 'rt'), header=0)
            except FileNotFoundError:
                continue
            # if df.shape[0] < 2**16 - 1:
            #     continue
            if algs_name not in algs_data:
                algs_data[algs_name] = [df['average_reward'].iloc[:front]]
                index = df.index.values[:front]
            else:
                algs_data[algs_name].append(df['average_reward'].iloc[:front])

    print(algs_data.keys())
    select = np.arange(0, len(index), step=2500)

    last_mean = []
    for file_name, algs_name, color in algs_plot_conf:
        tmp_list = algs_data[algs_name]
        # d = np.array(tmp_list)
        # d = np.vstack(d)
        d = np.asarray(tmp_list)
        mean = np.mean(d, axis=0)

        last_mean.append(mean[-1])
        # print("mean.shape", mean.shape)
        print(algs_name + "\t" + str(round(mean[-1],5)), end="\t")
        # print(round(aver_reward[-1], 4), end="\t")
        print(mean.shape[0], end="\t")
        print(file_name)
        std_err = np.std(d, axis=0)  / np.sqrt(len(d))
        tmp = plt.errorbar(index[select], mean[select], yerr=std_err[select], label=algs_name, color=color)
        l_list.append(tmp[0])
    improvement = cal_improve_from_index_last(last_mean)
    print("improvement\t", "\t".join(str(round(x*100, 2)) + "%" for x in improvement))

    plt.ylim(bottom=0)
    plt.title(dataset_name)
    plt.xlabel("Time t")
    plt.ylabel("Averaged Cumulative Reward")
    from matplotlib.ticker import FuncFormatter

    plt.gca().xaxis.set_major_formatter(FuncFormatter(x_update_scale_value))
    return l_list

for i, folder_datasetname_conf_each in enumerate(folder_datasetname_conf):
    folder_path, dataset_name, algs_plot_conf = folder_datasetname_conf_each
    l_list = draw(folder_path, dataset_name, algs_plot_conf, i)
line_labels = ['ConUCB', 'ConUCB (40%)', 'ConUCB (60%)', 'CtoF-ConUCB', 'CtoF-ConUCB-Clu', 'CtoF-ConUCB-Clu+']
fig.legend(l_list,     # The line objects
           labels=line_labels,   # The labels for each line
           loc="upper center",   # Position of legend
           # borderaxespad=0.1,    # Small spacing around legend box
           # title="Legend Title",  # Title for the legend
           ncol=6,
            prop={'size': 7}
           )
plt.subplots_adjust(wspace =0.3, hspace =0.35, top=0.84, bottom=0.08, left=0.1, right=0.95)
plt.savefig(f'{sig}.pdf')
plt.savefig(f'{sig}.png')
plt.show()

