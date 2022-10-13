screen -S 1006_n1_final
conda activate pytorch14_server1

cd /mnt/qzli_hdd/ConUCB/cluster_original/Clustering+ConUCB_exp
cd /data/qzli_hdd/ConUCB/cluster_original/Clustering+ConUCB_exp
cd /root/data/qzli/ConUCB/cluster_original/Clustering+ConUCB_exp/

CUDA_VISIBLE_DEVICES=3 nohup python __main__.py  --Env=Kmeans --alg=SKmeans  --para del+5 --T_ratio 0.01 --delta 10  --sclub_alpha 0.25 --alpha_p 1.4142135623730951 --alpha 0.1 --tilde_alpha 0.25  --lamb 0.3 --tilde_lamb 0.5 --seed=-1 --fn=../datasets/data3_movielens_generator/ml_25m_0401a1_un2000_in25000_0 --sig 1006_n1_final --N 16 --use_top_env --use_top_base --is_split_over_user_num --save_theta --save_key_term_embedding&
CUDA_VISIBLE_DEVICES=3 nohup python __main__.py  --Env=Kmeans --alg=Kmeans_user_ts_cluster  --para del+5 --delta 5 --alpha 0.1 --tilde_alpha 0.25  --lamb 0.3 --tilde_lamb 0.5 --user_ts_alpha 0.001 --seed=-1 --fn=../datasets/data3_movielens_generator/ml_25m_0401a1_un2000_in25000_0 --sig 1006_n1_final --N 16 --use_top_env --use_top_base --is_split_over_user_num --save_theta --save_key_term_embedding&

CUDA_VISIBLE_DEVICES=6 nohup python __main__.py  --Env=Kmeans --alg=Kmeans_user_ts_cluster  --para del+0.1 --delta 1 --alpha 0.01 --tilde_alpha 0.25  --lamb 0.5 --tilde_lamb 0.1 --user_ts_alpha 0.01 --seed=-1 --fn=../datasets/data4_lastfm/un_1500-bn_10000/train_item_user_test_rate --sig 1006_n1_final --N 16 --use_top_env --use_top_base --is_split_over_user_num --save_theta --save_key_term_embedding&
CUDA_VISIBLE_DEVICES=7 nohup python __main__.py  --Env=Kmeans --alg=SKmeans  --para del+0.1 --T_ratio 0.1 --delta 1  --sclub_alpha 0.25 --alpha_p 1.4142135623730951 --alpha 0.01 --tilde_alpha 0.25  --lamb 0.5 --tilde_lamb 0.1 --seed=-1 --fn=../datasets/data4_lastfm/un_1500-bn_10000/train_item_user_test_rate --sig 1006_n1_final --N 16 --use_top_env --use_top_base --is_split_over_user_num --save_theta --save_key_term_embedding&
