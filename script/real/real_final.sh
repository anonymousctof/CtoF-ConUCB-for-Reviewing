screen -S 1006_n1_final
conda activate pytorch14_server1

cd /mnt/qzli_hdd/ConUCB/cluster_original/Clustering+ConUCB_exp
cd /data/qzli_hdd/ConUCB/cluster_original/Clustering+ConUCB_exp
cd /root/data/qzli/ConUCB/cluster_original/Clustering+ConUCB_exp/
CUDA_VISIBLE_DEVICES=0 nohup python __main__.py --alpha 10 --lamb 0 --Env=Attr --alg=LinUCB --seed=-1 --fn=../datasets/data1_no_loc_tky_check_in/shuff_u_0623_un_1500-bn_5000/train_item_user_test_rate  --sig 1006_n1_final --N 16 --use_top_env --use_top_base&
CUDA_VISIBLE_DEVICES=0 nohup python __main__.py --alpha 10 --lamb 0.5 --Env=Attr --alg=ArmCon  --seed=-1 --fn=../datasets/data1_no_loc_tky_check_in/shuff_u_0623_un_1500-bn_5000/train_item_user_test_rate --sig 1006_n1_final --N 16 --use_top_env --use_top_base &
CUDA_VISIBLE_DEVICES=0 nohup python __main__.py --alpha 0.25 --tilde_alpha 10 --lamb 0.5 --tilde_lamb 0.7 --Env=Attr --alg=ConUCB --seed=-1 --fn=../datasets/data1_no_loc_tky_check_in/shuff_u_0623_un_1500-bn_5000/train_item_user_test_rate --sig 1006_n1_final --N 16 --use_top_env --use_top_base &
CUDA_VISIBLE_DEVICES=0 nohup python __main__.py --Env=Kmeans --alg=ConUCB --para del+10 --alpha 0.1 --lamb 0.7 --tilde_alpha 10 --tilde_lamb 0.5 --seed=-1 --fn=../datasets/data1_no_loc_tky_check_in/shuff_u_0623_un_1500-bn_5000/train_item_user_test_rate --sig 1006_n1_final --N 16 --use_top_env --use_top_base &
CUDA_VISIBLE_DEVICES=0 nohup python __main__.py  --Env=Kmeans --alg=SKmeans  --para del+10 --T_ratio 0.1 --delta 0.1  --sclub_alpha 1 --alpha_p 1.4142135623730951 --alpha 0.1 --tilde_alpha 2  --lamb 0.5 --tilde_lamb 0.7 --seed=-1 --fn=../datasets/data1_no_loc_tky_check_in/shuff_u_0623_un_1500-bn_5000/train_item_user_test_rate --sig 1006_n1_final --N 16 --use_top_env --use_top_base --is_split_over_user_num&
CUDA_VISIBLE_DEVICES=0 nohup python __main__.py  --Env=Kmeans --alg=Kmeans_user_ts_cluster  --para del+10 --delta 0.1 --alpha 0.1 --tilde_alpha 10  --lamb 0.5 --tilde_lamb 0.7 --user_ts_alpha 10 --seed=-1 --fn=../datasets/data1_no_loc_tky_check_in/shuff_u_0623_un_1500-bn_5000/train_item_user_test_rate --sig 1006_n1_final --N 16 --use_top_env --use_top_base --is_split_over_user_num&


CUDA_VISIBLE_DEVICES=3 nohup python __main__.py --alpha 10 --lamb 0 --Env=Attr --alg=LinUCB --seed=-1 --fn=../datasets/data2_delicious_KONECT/shuff_u_0619_un_1500-bn_5000/train_item_user_test_rate  --sig 1006_n1_final --N 16 --use_top_env --use_top_base&
CUDA_VISIBLE_DEVICES=3 nohup python __main__.py --alpha 10 --lamb 0.5 --Env=Attr --alg=ArmCon  --seed=-1 --fn=../datasets/data2_delicious_KONECT/shuff_u_0619_un_1500-bn_5000/train_item_user_test_rate --sig 1006_n1_final --N 16 --use_top_env --use_top_base &
CUDA_VISIBLE_DEVICES=3 nohup python __main__.py --alpha 10 --tilde_alpha 0.25 --lamb 0.5 --tilde_lamb 1 --Env=Attr --alg=ConUCB --seed=-1 --fn=../datasets/data2_delicious_KONECT/shuff_u_0619_un_1500-bn_5000/train_item_user_test_rate --sig 1006_n1_final --N 16 --use_top_env --use_top_base &
CUDA_VISIBLE_DEVICES=3 nohup python __main__.py --Env=Kmeans --alg=ConUCB --para del+1 --alpha 10 --lamb 0.5 --tilde_alpha 1 --tilde_lamb 1 --seed=-1 --fn=../datasets/data2_delicious_KONECT/shuff_u_0619_un_1500-bn_5000/train_item_user_test_rate --sig 1006_n1_final --N 16 --use_top_env --use_top_base &
CUDA_VISIBLE_DEVICES=3 nohup python __main__.py  --Env=Kmeans --alg=SKmeans  --para del+0.5 --T_ratio 0.1 --delta 0.1  --sclub_alpha 0.1 --alpha_p 1.4142135623730951 --alpha 10 --tilde_alpha 1  --lamb 0.5 --tilde_lamb 1 --seed=-1 --fn=../datasets/data2_delicious_KONECT/shuff_u_0619_un_1500-bn_5000/train_item_user_test_rate --sig 1006_n1_final --N 16 --use_top_env --use_top_base --is_split_over_user_num&
CUDA_VISIBLE_DEVICES=3 nohup python __main__.py  --Env=Kmeans --alg=Kmeans_user_ts_cluster  --para del+0.5 --delta 0.1 --alpha 10 --tilde_alpha 1  --lamb 0.5 --tilde_lamb 1 --user_ts_alpha 0.01 --seed=-1 --fn=../datasets/data2_delicious_KONECT/shuff_u_0619_un_1500-bn_5000/train_item_user_test_rate --sig 1006_n1_final --N 16 --use_top_env --use_top_base --is_split_over_user_num&


CUDA_VISIBLE_DEVICES=0 nohup python __main__.py --alpha 0.1 --lamb 0 --Env=Attr --alg=LinUCB --seed=-1 --fn=../datasets/data3_movielens_generator/ml_25m_0401a1_un2000_in25000_0  --sig 1006_n1_final --N 16 --use_top_env --use_top_base&
CUDA_VISIBLE_DEVICES=0 nohup python __main__.py --alpha 0.1 --lamb 0.5 --Env=Attr --alg=ArmCon  --seed=-1 --fn=../datasets/data3_movielens_generator/ml_25m_0401a1_un2000_in25000_0 --sig 1006_n1_final --N 16 --use_top_env --use_top_base &
CUDA_VISIBLE_DEVICES=1 nohup python __main__.py --alpha 0.01 --tilde_alpha 0.25 --lamb 0.5 --tilde_lamb 0.9 --Env=Attr --alg=ConUCB --seed=-1 --fn=../datasets/data3_movielens_generator/ml_25m_0401a1_un2000_in25000_0 --sig 1006_n1_final --N 16 --use_top_env --use_top_base &
CUDA_VISIBLE_DEVICES=1 nohup python __main__.py --Env=Kmeans --alg=ConUCB --para del+5 --alpha 0.1 --lamb 0.3 --tilde_alpha 0.25 --tilde_lamb 0.5 --seed=-1 --fn=../datasets/data3_movielens_generator/ml_25m_0401a1_un2000_in25000_0 --sig 1006_n1_final --N 16 --use_top_env --use_top_base &
CUDA_VISIBLE_DEVICES=3 nohup python __main__.py  --Env=Kmeans --alg=SKmeans  --para del+5 --T_ratio 0.01 --delta 10  --sclub_alpha 0.25 --alpha_p 1.4142135623730951 --alpha 0.1 --tilde_alpha 0.25  --lamb 0.3 --tilde_lamb 0.5 --seed=-1 --fn=../datasets/data3_movielens_generator/ml_25m_0401a1_un2000_in25000_0 --sig 1006_n1_final --N 16 --use_top_env --use_top_base --is_split_over_user_num & # &--save_theta --save_key_term_embedding&
CUDA_VISIBLE_DEVICES=3 nohup python __main__.py  --Env=Kmeans --alg=Kmeans_user_ts_cluster  --para del+5 --delta 5 --alpha 0.1 --tilde_alpha 0.25  --lamb 0.3 --tilde_lamb 0.5 --user_ts_alpha 0.001 --seed=-1 --fn=../datasets/data3_movielens_generator/ml_25m_0401a1_un2000_in25000_0 --sig 1006_n1_final --N 16 --use_top_env --use_top_base --is_split_over_user_num& #  --save_theta --save_key_term_embedding&



CUDA_VISIBLE_DEVICES=5 nohup python __main__.py --alpha 1 --lamb 0 --Env=Attr --alg=LinUCB --seed=-1 --fn=../datasets/data4_lastfm/un_1500-bn_10000/train_item_user_test_rate  --sig 1006_n1_final --N 16 --use_top_env --use_top_base&
CUDA_VISIBLE_DEVICES=5 nohup python __main__.py --alpha 0.1 --lamb 0.5 --Env=Attr --alg=ArmCon  --seed=-1 --fn=../datasets/data4_lastfm/un_1500-bn_10000/train_item_user_test_rate --sig 1006_n1_final --N 16 --use_top_env --use_top_base &
CUDA_VISIBLE_DEVICES=5 nohup python __main__.py --alpha 1 --tilde_alpha 0.25 --lamb 0.5 --tilde_lamb 0.1 --Env=Attr --alg=ConUCB --seed=-1 --fn=../datasets/data4_lastfm/un_1500-bn_10000/train_item_user_test_rate --sig 1006_n1_final --N 16 --use_top_env --use_top_base &
CUDA_VISIBLE_DEVICES=5 nohup python __main__.py --Env=Kmeans --alg=ConUCB --para del+0.1 --alpha 0.01 --lamb 0.9 --tilde_alpha 0.25 --tilde_lamb 0.1 --seed=-1 --fn=../datasets/data4_lastfm/un_1500-bn_10000/train_item_user_test_rate --sig 1006_n1_final --N 16 --use_top_env --use_top_base &
CUDA_VISIBLE_DEVICES=6 nohup python __main__.py  --Env=Kmeans --alg=Kmeans_user_ts_cluster  --para del+0.1 --delta 1 --alpha 0.01 --tilde_alpha 0.25  --lamb 0.5 --tilde_lamb 0.1 --user_ts_alpha 0.01 --seed=-1 --fn=../datasets/data4_lastfm/un_1500-bn_10000/train_item_user_test_rate --sig 1006_n1_final --N 16 --use_top_env --use_top_base --is_split_over_user_num& #  --save_theta --save_key_term_embedding&
CUDA_VISIBLE_DEVICES=7 nohup python __main__.py  --Env=Kmeans --alg=SKmeans  --para del+0.1 --T_ratio 0.1 --delta 1  --sclub_alpha 0.25 --alpha_p 1.4142135623730951 --alpha 0.01 --tilde_alpha 0.25  --lamb 0.5 --tilde_lamb 0.1 --seed=-1 --fn=../datasets/data4_lastfm/un_1500-bn_10000/train_item_user_test_rate --sig 1006_n1_final --N 16 --use_top_env --use_top_base --is_split_over_user_num --save_theta --save_key_term_embedding&


CUDA_VISIBLE_DEVICES=7 nohup python __main__.py --alpha 0.25 --lamb 0 --Env=Attr --alg=LinUCB --seed=-1 --fn=../datasets/data5_bibsonomy/un_1500-bn_5000/train_item_user_test_rate  --sig 1006_n1_final --N 16 --use_top_env --use_top_base&
CUDA_VISIBLE_DEVICES=7 nohup python __main__.py --alpha 0.1 --lamb 0.5 --Env=Attr --alg=ArmCon  --seed=-1 --fn=../datasets/data5_bibsonomy/un_1500-bn_5000/train_item_user_test_rate --sig 1006_n1_final --N 16 --use_top_env --use_top_base &
CUDA_VISIBLE_DEVICES=7 nohup python __main__.py --alpha 0.25 --tilde_alpha 0.5 --lamb 0.5 --tilde_lamb 0.7 --Env=Attr --alg=ConUCB --seed=-1 --fn=../datasets/data5_bibsonomy/un_1500-bn_5000/train_item_user_test_rate --sig 1006_n1_final --N 16 --use_top_env --use_top_base &
CUDA_VISIBLE_DEVICES=7 nohup python __main__.py --Env=Kmeans --alg=ConUCB --para del+50 --alpha 0.5 --lamb 0.5 --tilde_alpha 0.5 --tilde_lamb 0.7 --seed=-1 --fn=../datasets/data5_bibsonomy/un_1500-bn_5000/train_item_user_test_rate --sig 1006_n1_final --N 16 --use_top_env --use_top_base &
CUDA_VISIBLE_DEVICES=7 nohup python __main__.py  --Env=Kmeans --alg=SKmeans  --para del+50 --T_ratio 0.01 --delta 1000 --sclub_alpha 10 --alpha_p 1.4142135623730951 --alpha 0.5 --tilde_alpha 0.5  --lamb 0.5 --tilde_lamb 0.7 --seed=-1 --fn=../datasets/data5_bibsonomy/un_1500-bn_5000/train_item_user_test_rate --sig 1006_n1_final --N 16 --use_top_env --use_top_base --is_split_over_user_num&
CUDA_VISIBLE_DEVICES=7 nohup python __main__.py  --Env=Kmeans --alg=Kmeans_user_ts_cluster  --para del+50 --delta 1000 --alpha 0.5 --tilde_alpha 0.5  --lamb 0.5 --tilde_lamb 0.7 --user_ts_alpha 0.1 --seed=-1 --fn=../datasets/data5_bibsonomy/un_1500-bn_5000/train_item_user_test_rate --sig 1006_n1_final --N 16 --use_top_env --use_top_base --is_split_over_user_num&


CUDA_VISIBLE_DEVICES=0 nohup python __main__.py --alpha 0.1 --lamb 0 --Env=Attr --alg=LinUCB --seed=-1 --fn=../datasets/data6_visualizeus/un_1500-bn_10000/train_item_user_test_rate  --sig 1006_n1_final --N 16 --use_top_env --use_top_base&
CUDA_VISIBLE_DEVICES=0 nohup python __main__.py --alpha 0.1 --lamb 0.5 --Env=Attr --alg=ArmCon  --seed=-1 --fn=../datasets/data6_visualizeus/un_1500-bn_10000/train_item_user_test_rate --sig 1006_n1_final --N 16 --use_top_env --use_top_base &
CUDA_VISIBLE_DEVICES=3 nohup python __main__.py --alpha 0.25 --tilde_alpha 0.1 --lamb 0.5 --tilde_lamb 0.3 --Env=Attr --alg=ConUCB --seed=-1 --fn=../datasets/data6_visualizeus/un_1500-bn_10000/train_item_user_test_rate --sig 1006_n1_final --N 16 --use_top_env --use_top_base &
CUDA_VISIBLE_DEVICES=3 nohup python __main__.py --Env=Kmeans --alg=ConUCB --para del+5 --alpha 0.01 --lamb 0.5 --tilde_alpha 0.1 --tilde_lamb 0.3 --seed=-1 --fn=../datasets/data6_visualizeus/un_1500-bn_10000/train_item_user_test_rate --sig 1006_n1_final --N 16 --use_top_env --use_top_base &
CUDA_VISIBLE_DEVICES=5 nohup python __main__.py  --Env=Kmeans --alg=SKmeans  --para del+10 --T_ratio 0.1 --delta 10  --sclub_alpha 1 --alpha_p 1.4142135623730951 --alpha 0.1 --tilde_alpha 0.1  --lamb 0.5 --tilde_lamb 0.3 --seed=-1 --fn=../datasets/data6_visualizeus/un_1500-bn_10000/train_item_user_test_rate --sig 1006_n1_final --N 16 --use_top_env --use_top_base --is_split_over_user_num&
CUDA_VISIBLE_DEVICES=5 nohup python __main__.py  --Env=Kmeans --alg=Kmeans_user_ts_cluster  --para del+10 --delta 1000 --alpha 0.1 --tilde_alpha 0.1  --lamb 0.5 --tilde_lamb 0.3 --user_ts_alpha 0.001 --seed=-1 --fn=../datasets/data6_visualizeus/un_1500-bn_10000/train_item_user_test_rate --sig 1006_n1_final --N 16 --use_top_env --use_top_base --is_split_over_user_num&
