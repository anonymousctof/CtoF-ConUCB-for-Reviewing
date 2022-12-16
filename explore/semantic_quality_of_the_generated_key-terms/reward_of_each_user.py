import numpy as np
# from wordcloud import generate_from_frequencies
import matplotlib.pyplot as plt
from wordcloud import WordCloud,ImageColorGenerator

import pandas as pd
run_sig = "1006_n1_final"

sig = "1213"

file_part_name = "results_r.csv"
dataset_name = 'MovieLens'
alg = "CtoF-ConUCB-Clu+"
path = f"../../dataset-generate/data4_movielens_generator/ml_25m_0401a1_un2000_in25000_0/{run_sig}/0/[Kmeans+del+5]+topk_[Kmeans_user_ts_cluster+Dyn+5.0+kalpha+0.1+ktalpha+0.25+lamb0.3+tlamb0.5+user_ts_a0.001+is_split_over_user_num]"

file_part_name = "results_0622.csv"
dataset_name = "LastFM"
alg = "CtoF-ConUCB-Clu+"
path = f"../../dataset-generate/data5_lastfm/un_1500-bn_10000/train_item_user_test_rate/{run_sig}/0/[Kmeans+del+0.1]+topk_[Kmeans_user_ts_cluster+Dyn+1.0+kalpha+0.01+ktalpha+0.25+lamb0.5+tlamb0.1+user_ts_a0.01+is_split_over_user_num]"

file_save = f"{sig}_{dataset_name}_{alg}/"
import os
os.makedirs(file_save, exist_ok=True)
def draw(path, file_save):
    import gzip
    df = pd.read_csv(gzip.open(path + ".gz"), header=0)
    print(df.columns)
    reward_df = pd.DataFrame(columns = ['user_id', 'user_aver_reward', 'recommend_times'])
    for i in range(500):
        df_user = df[df['user_id'] == i]
        df_average_reward = np.mean(df_user['user_reward'])
        reward_df.loc[len(reward_df.index)] = [i, df_average_reward, df_user.shape[0]]
    reward_df.to_csv(f"{file_save}/0_reward.csv", index=False)
    reward_df = reward_df.sort_values("user_aver_reward", ascending=False)
    reward_df.to_csv(f"{file_save}/0_reward_sorted.csv", index=False)
    print(file_save)
    # print(reward_df.head(5))
    # print(reward_df.tail(5))


draw(path, file_save)
