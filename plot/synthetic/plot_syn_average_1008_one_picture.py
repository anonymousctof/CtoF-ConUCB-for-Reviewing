
import numpy as np
import pandas as pd
import os
import gzip
import seaborn as sns
import matplotlib.pyplot as plt
data = dict()
front = 2**17 - 1
sig = "1013_syn_final"
c1 = (31/255, 119/255, 180/255)
c2 = (139/255, 139/255, 139/255)
c3 = (38/255, 217/255, 175/255)
c3 = (38/255, 217/255, 175/255)
c4 = (125/255, 46/255, 140/255)
c5 = (237/255, 176/255, 30/255)
c6 = (214/255, 39/255, 39/255)
folder_datasetname_conf = []
folder_path = "./datasets/synthetic_datasets/con_data_50_0.05_1/1013_syn_final"
dataset_name = "(a) $\sigma_k = 1.00$"
algs_plot_conf =[
                ('[Item+1.0]+topk_[LinUCB+alpha+0.25+lamb0.0+talpha0.25+tlamb1.0].gz', 'LinUCB', c1),
                ('[Item+1.0]+topk_[ArmCon+alpha+0.1+lamb0.5+talpha0.25+tlamb1.0].gz', 'ArmCon', c2),
                ('[Item+1.0]+topk_[ConUCB+alpha+0.25+lamb0.5+talpha0.1+tlamb1.0]-selection.gz', 'ConUCB', c3),
                ('[Kmeans+del+10]+topk_[ConUCB+alpha+0.25+lamb0.5+talpha0.1+tlamb1.0]-no_selection.gz', 'CtoF-ConUCB', c4),
                ('[Kmeans+del+10.0]++T_ratio0.2topk_[SKmeans+Dyn+0.01+salpha+0.5+salpha_p+1.4142135623730951+kalpha+0.25+ktalpha+0.1+lamb0.5+tlamb1.0+T_ratio0.2+is_split_over_user_num].gz', 'CtoF-ConUCB-Clu', c5),
                ('[Kmeans+del+10.0]+topk_[Kmeans_user_ts_cluster+Dyn+0.01+kalpha+0.01+ktalpha+0.1+lamb0.5+tlamb0.1+user_ts_a0.001+is_split_over_user_num].gz', 'CtoF-ConUCB-Clu+', c6),
                ]
folder_datasetname_conf.append((folder_path, dataset_name, algs_plot_conf))
folder_path = "./datasets/synthetic_datasets/con_data_50_0.05_10/1013_syn_final"
dataset_name = "(b) $\sigma_k = 10.00$"
algs_plot_conf =[
                ('[Item+1.0]+topk_[LinUCB+alpha+0.25+lamb0.0+talpha0.25+tlamb1.0].gz', 'LinUCB', c1),
                ('[Item+1.0]+topk_[ArmCon+alpha+0.25+lamb0.5+talpha0.25+tlamb1.0].gz', 'ArmCon', c2),
                ('[Item+1.0]+topk_[ConUCB+alpha+0.25+lamb0.5+talpha0.1+tlamb0.9]-selection.gz', 'ConUCB', c3),
                ('[Kmeans+del+50]+topk_[ConUCB+alpha+0.25+lamb0.5+talpha0.1+tlamb0.9]-no_selection.gz', 'CtoF-ConUCB', c4),
                ('[Kmeans+del+50.0]++T_ratio0.1topk_[SKmeans+Dyn+0.01+salpha+0.5+salpha_p+1.4142135623730951+kalpha+0.25+ktalpha+0.1+lamb0.5+tlamb0.9+T_ratio0.1+is_split_over_user_num].gz', 'CtoF-ConUCB-Clu', c5),
                ('[Kmeans+del+50.0]+topk_[Kmeans_user_ts_cluster+Dyn+0.01+kalpha+0.01+ktalpha+0.01+lamb0.5+tlamb0.9+user_ts_a0.001+is_split_over_user_num].gz', 'CtoF-ConUCB-Clu+', c6),
                ]
folder_datasetname_conf.append((folder_path, dataset_name, algs_plot_conf))

folder_path = "./datasets/synthetic_datasets/con_data_50_0.01_5/1013_syn_final"
dataset_name = "(c) $\sigma_u = 0.01$"
algs_plot_conf =[
                ('[Item+1.0]+topk_[LinUCB+alpha+0.25+lamb0.0+talpha0.25+tlamb1.0].gz', 'LinUCB', c1),
                ('[Item+1.0]+topk_[ArmCon+alpha+0.25+lamb0.5+talpha0.25+tlamb1.0].gz', 'ArmCon', c2),
                ('[Item+1.0]+topk_[ConUCB+alpha+0.01+lamb0.5+talpha0.25+tlamb1.0]-selection.gz', 'ConUCB', c3),
                ('[Kmeans+del+100]+topk_[ConUCB+alpha+0.01+lamb0.5+talpha0.25+tlamb1.0]-no_selection.gz', 'CtoF-ConUCB', c4),
                ('[Kmeans+del+100.0]++T_ratio0.1topk_[SKmeans+Dyn+0.1+salpha+0.5+salpha_p+1.4142135623730951+kalpha+0.01+ktalpha+0.25+lamb0.5+tlamb1.0+T_ratio0.1+is_split_over_user_num].gz', 'CtoF-ConUCB-Clu', c5),
                ('[Kmeans+del+100.0]+topk_[Kmeans_user_ts_cluster+Dyn+0.1+kalpha+0.01+ktalpha+0.25+lamb0.5+tlamb1.0+user_ts_a0.001+is_split_over_user_num].gz', 'CtoF-ConUCB-Clu+', c6),
                ]
folder_datasetname_conf.append((folder_path, dataset_name, algs_plot_conf))

folder_path = "./datasets/synthetic_datasets/con_data_50_0.1_5/1013_syn_final"
dataset_name = "(d) $\sigma_u = 0.10$"
algs_plot_conf =[
                ('[Item+1.0]+topk_[LinUCB+alpha+0.25+lamb0.0+talpha0.25+tlamb1.0].gz', 'LinUCB', c1),
                ('[Item+1.0]+topk_[ArmCon+alpha+0.25+lamb0.5+talpha0.25+tlamb1.0].gz', 'ArmCon', c2),
                ('[Item+1.0]+topk_[ConUCB+alpha+0.25+lamb0.5+talpha0.1+tlamb0.9]-selection.gz', 'ConUCB', c3),
                ('[Kmeans+del+50]+topk_[ConUCB+alpha+0.25+lamb0.5+talpha0.1+tlamb0.9]-no_selection.gz', 'CtoF-ConUCB', c4),
                ('[Kmeans+del+50.0]++T_ratio0.1topk_[SKmeans+Dyn+0.01+salpha+0.5+salpha_p+1.4142135623730951+kalpha+0.25+ktalpha+0.1+lamb0.5+tlamb0.9+T_ratio0.1+is_split_over_user_num].gz', 'CtoF-ConUCB-Clu', c5),
                ('[Kmeans+del+50.0]+topk_[Kmeans_user_ts_cluster+Dyn+10.0+kalpha+0.25+ktalpha+0.1+lamb0.5+tlamb0.9+user_ts_a0.001+is_split_over_user_num].gz', 'CtoF-ConUCB-Clu+', c6),
                ]
folder_datasetname_conf.append((folder_path, dataset_name, algs_plot_conf))
print(folder_datasetname_conf)

dataset_name_to_save = "-".join(folder_path.strip().split('/')[-2:-1])


def x_update_scale_value(temp, position):
    result = temp // 1000
    return "{}k".format(int(result))

x = np.array([1, 2, 3, 4])
y = np.array([1, 4, 9, 16])
fig = plt.figure(figsize=(7.5, 7.6))
ax_list = fig.subplots(nrows=2, ncols=2)

sns.set(style="whitegrid")
def draw(folder_path, dataset_name, algs_plot_conf, location_subplot):
    ax = plt.subplot(2, 2, location_subplot+1)
    l_list = []

    algs_data = {}
    for seed in range(10):
        for file_name, algs_name, color in algs_plot_conf:
            file_path = f"{folder_path}/{seed}/{file_name}"
            try:
                df = pd.read_csv(gzip.open(file_path, 'rt'), header=0)
            except FileNotFoundError:
                print("not exist", file_path)
                continue
            if algs_name not in algs_data:
                algs_data[algs_name] = [df['average_reward'].iloc[:front]]
                index = df.index.values[:front]
            else:
                algs_data[algs_name].append(df['average_reward'].iloc[:front])
            if algs_data[algs_name][-1].shape[0] < front:
                print(f"algs_name: {algs_name}", algs_data[algs_name][-1].shape[0])

    print(algs_data.keys())
    select = np.arange(0, len(index), step=2500)

    last_mean = []
    for file_name, algs_name, color in algs_plot_conf:
        tmp_list = algs_data[algs_name]
        d = np.asarray(tmp_list)
        # print(d.shape)
        mean = np.mean(d, axis=0)
        last_mean.append(mean[-1])
        print(algs_name + "\t" + str(round(mean[-1],5)), end="\t")
        print(mean.shape[0], end="\t")
        print(file_name)
        std_err = np.std(d, axis=0) / np.sqrt(len(d))
        tmp = plt.errorbar(index[select], mean[select], yerr=std_err[select], label=algs_name, color=color)
        l_list.append(tmp[0])


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
line_labels = ['LinUCB', 'ArmCon', 'ConUCB', 'CtoF-ConUCB', 'CtoF-ConUCB-Clu', 'CtoF-ConUCB-Clu+']
fig.legend(l_list,     # The line objects
           labels=line_labels,   # The labels for each line
           loc="upper center",   # Position of legend
           # borderaxespad=0.1,    # Small spacing around legend box
           # title="Legend Title",  # Title for the legend
           ncol=6,
            prop={'size': 8}
           )
plt.subplots_adjust(wspace =0.3, hspace =0.35, top=0.92, bottom=0.08, left=0.1, right=0.95)
plt.savefig(f'{sig}.pdf')
plt.savefig(f'{sig}.png')
plt.show()

