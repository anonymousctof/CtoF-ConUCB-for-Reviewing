import numpy as np
# from wordcloud import generate_from_frequencies
import matplotlib.pyplot as plt
from wordcloud import WordCloud,ImageColorGenerator

import pandas as pd

sig = "1213"
sig_sort_file = "1213"


# file_part_name = "results_r.csv"
# dataset_name = 'MovieLens'
# alg = "CtoF-ConUCB-Clu+"
# path = f"../../dataset-generate/data4_movielens_generator/ml_25m_0401a1_un2000_in25000_0/1008_save_theta_cluster/0/[Kmeans+del+5]+topk_[Kmeans_user_ts_cluster+Dyn+5.0+kalpha+0.1+ktalpha+0.25+lamb0.3+tlamb0.5+user_ts_a0.001+is_split_over_user_num]"

file_part_name = "results_0622.csv"
dataset_name = "LastFM"
alg = "CtoF-ConUCB-Clu+"
path = f"../../dataset-generate/data5_lastfm/un_1500-bn_10000/train_item_user_test_rate/1008_save_theta_cluster/0/[Kmeans+del+0.1]+topk_[Kmeans_user_ts_cluster+Dyn+1.0+kalpha+0.01+ktalpha+0.25+lamb0.5+tlamb0.1+user_ts_a0.01+is_split_over_user_num]"

sorted_file_save = f"{sig_sort_file}_{dataset_name}_{alg}/"
file_save = f"{sig}_{dataset_name}_{alg}/"
import os
os.makedirs(file_save, exist_ok=True)

pd.set_option('display.max_columns', None)
def read_time_index_key_term(path_word):
    df = pd.read_csv(path_word, sep='\t', header = None, skiprows=1)
    columns = ['time', 'index', 'attr_semantic1', 'attr_semantic2', 'attr_semantic3', 'attr_semantic4', 'attr_semantic5', 'items', 'user', 'reward']
    df.columns = columns
    semantic_all_list = []
    semantic_all_list.append(df['attr_semantic1'].values)
    semantic_all = np.concatenate(semantic_all_list)
    fq = pd.value_counts(semantic_all)
    fq_dict = {}
    for index in fq.index:
        fq_dict[index] = fq.loc[index]
    return fq_dict


df = pd.read_csv(f"{sorted_file_save}/0_reward_sorted.csv")
user_list_top = df.head(3)['user_id']
user_list_low = df.tail(3)['user_id']
print("64 user_list_low", user_list_low)

user_list_low_new = []
print(len(user_list_low))
len_user_list_low = len(user_list_low)
for i in range(len_user_list_low):
    user_list_low_new.append(user_list_low.iloc[len_user_list_low - 1 - i])
user_list_low = user_list_low_new

def plot_word_cloud(user_id, file_save_top_low):
    print(user_id)
    path_word = path + f"/semantic/user_{user_id}/{file_part_name}"
    fq = read_time_index_key_term(path_word)

    for i in fq.keys():
        print(i)
    a = max(fq.values())
    wc = WordCloud(relative_scaling=1, max_font_size=a * 20, background_color='white')  # , colormap='RdBu')
    wc.generate_from_frequencies(fq)

    wc_svg = wc.to_svg(embed_font=True)
    with open(file_save_top_low + f'user_{user_id}.svg', 'w+') as f:
        f.write(wc_svg)

file_save_move_low = file_save + f"/select_low_{sig}/"
os.makedirs(file_save_move_low, exist_ok=True)
for user_id in user_list_low:
    user_id = round(user_id)
    plot_word_cloud(user_id, file_save_move_low)


file_save_move_top = file_save + f"/select_top_{sig}/"
os.makedirs(file_save_move_top, exist_ok=True)
for user_id in user_list_top:
    user_id = round(user_id)
    plot_word_cloud(user_id, file_save_move_top)
