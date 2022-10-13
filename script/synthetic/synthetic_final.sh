

screen -S 1013_syn_final
conda activate pytorch14_server1
cd ./Clustering+ConUCB_exp/

CUDA_VISIBLE_DEVICES=0 nohup python __main__.py --alpha 0.25 --lamb 0 --Env=Attr --alg=LinUCB --seed=-1 --fn=../datasets/synthetic_datasets/con_data_50_0.05_1/  --sig 1013_syn_final --N 17 --use_top_env --use_top_base&
CUDA_VISIBLE_DEVICES=1 nohup python __main__.py --alpha 0.25 --lamb 0 --Env=Attr --alg=LinUCB --seed=-1 --fn=../datasets/synthetic_datasets/con_data_50_0.05_10/  --sig 1013_syn_final --N 17 --use_top_env --use_top_base&
CUDA_VISIBLE_DEVICES=2 nohup python __main__.py --alpha 0.25 --lamb 0 --Env=Attr --alg=LinUCB --seed=-1 --fn=../datasets/synthetic_datasets/con_data_50_0.01_5/  --sig 1013_syn_final --N 17 --use_top_env --use_top_base&
CUDA_VISIBLE_DEVICES=3 nohup python __main__.py --alpha 0.25 --lamb 0 --Env=Attr --alg=LinUCB --seed=-1 --fn=../datasets/synthetic_datasets/con_data_50_0.1_5/  --sig 1013_syn_final --N 17 --use_top_env --use_top_base&

CUDA_VISIBLE_DEVICES=0 nohup python __main__.py --alpha 0.10 --lamb 0.5 --Env=Attr --alg=ArmCon --seed=-1 --fn=../datasets/synthetic_datasets/con_data_50_0.05_1/  --sig 1013_syn_final --N 17 --use_top_env --use_top_base&
CUDA_VISIBLE_DEVICES=1 nohup python __main__.py --alpha 0.25 --lamb 0.5 --Env=Attr --alg=ArmCon --seed=-1 --fn=../datasets/synthetic_datasets/con_data_50_0.05_10/  --sig 1013_syn_final --N 17 --use_top_env --use_top_base&
CUDA_VISIBLE_DEVICES=2 nohup python __main__.py --alpha 0.25 --lamb 0.5 --Env=Attr --alg=ArmCon --seed=-1 --fn=../datasets/synthetic_datasets/con_data_50_0.01_5/  --sig 1013_syn_final --N 17 --use_top_env --use_top_base&
CUDA_VISIBLE_DEVICES=3 nohup python __main__.py --alpha 0.25 --lamb 0.5 --Env=Attr --alg=ArmCon --seed=-1 --fn=../datasets/synthetic_datasets/con_data_50_0.1_5/  --sig 1013_syn_final --N 17 --use_top_env --use_top_base&

CUDA_VISIBLE_DEVICES=0 nohup python __main__.py --alpha 0.25 --tilde_alpha 0.1 --lamb 0.5 --tilde_lamb 1 --Env=Attr --alg=ConUCB --seed=-1 --fn=../datasets/synthetic_datasets/con_data_50_0.05_1/ --sig 1013_syn_final --N 17 --use_top_env --use_top_base &
CUDA_VISIBLE_DEVICES=1 nohup python __main__.py --alpha 0.25 --tilde_alpha 0.1 --lamb 0.5 --tilde_lamb 0.9 --Env=Attr --alg=ConUCB --seed=-1 --fn=../datasets/synthetic_datasets/con_data_50_0.05_10/ --sig 1013_syn_final --N 17 --use_top_env --use_top_base &
CUDA_VISIBLE_DEVICES=2 nohup python __main__.py --alpha 0.01 --tilde_alpha 0.25 --lamb 0.5 --tilde_lamb 1 --Env=Attr --alg=ConUCB --seed=-1 --fn=../datasets/synthetic_datasets/con_data_50_0.01_5/ --sig 1013_syn_final --N 17 --use_top_env --use_top_base &
CUDA_VISIBLE_DEVICES=3 nohup python __main__.py --alpha 0.25 --tilde_alpha 0.1 --lamb 0.5 --tilde_lamb 0.9 --Env=Attr --alg=ConUCB --seed=-1 --fn=../datasets/synthetic_datasets/con_data_50_0.1_5/ --sig 1013_syn_final --N 17 --use_top_env --use_top_base &

CUDA_VISIBLE_DEVICES=0 nohup python __main__.py --para del+10 --alpha 0.25 --lamb 0.5 --tilde_alpha 0.1 --tilde_lamb 1.0 --Env=Kmeans --alg=ConUCB --seed=-1 --fn=../datasets/synthetic_datasets/con_data_50_0.05_1/ --sig 1013_syn_final --N 17 --use_top_env --use_top_base  --no_use_selection&
CUDA_VISIBLE_DEVICES=1 nohup python __main__.py --para del+50 --alpha 0.25 --lamb 0.5 --tilde_alpha 0.1 --tilde_lamb 0.9 --Env=Kmeans --alg=ConUCB --seed=-1 --fn=../datasets/synthetic_datasets/con_data_50_0.05_10/ --sig 1013_syn_final --N 17 --use_top_env --use_top_base  --no_use_selection&
CUDA_VISIBLE_DEVICES=2 nohup python __main__.py --para del+100 --alpha 0.01 --lamb 0.5 --tilde_alpha 0.25 --tilde_lamb 1.0 --Env=Kmeans --alg=ConUCB --seed=-1 --fn=../datasets/synthetic_datasets/con_data_50_0.01_5/ --sig 1013_syn_final --N 17 --use_top_env --use_top_base  --no_use_selection&
CUDA_VISIBLE_DEVICES=3 nohup python __main__.py --para del+50 --alpha 0.25 --lamb 0.5 --tilde_alpha 0.1 --tilde_lamb 0.9 --Env=Kmeans --alg=ConUCB --seed=-1 --fn=../datasets/synthetic_datasets/con_data_50_0.1_5/ --sig 1013_syn_final --N 17 --use_top_env --use_top_base  --no_use_selection&

CUDA_VISIBLE_DEVICES=0 nohup python __main__.py --para --T_ratio 0.2 del+10.0 --delta 0.01 --alpha 0.25 --lamb 0.5 --tilde_alpha 0.1 --tilde_lamb 1.0 --Env=Kmeans --alg=SKmeans --seed=-1 --fn=../datasets/synthetic_datasets/con_data_50_0.05_1/ --sig 1013_syn_final --N 17 --use_top_env --use_top_base  --no_use_selection --is_split_over_user_num&
CUDA_VISIBLE_DEVICES=1 nohup python __main__.py --para --T_ratio 0.1 del+50.0 --delta 0.01 --alpha 0.25 --lamb 0.5 --tilde_alpha 0.1 --tilde_lamb 0.9 --Env=Kmeans --alg=SKmeans --seed=-1 --fn=../datasets/synthetic_datasets/con_data_50_0.05_10/ --sig 1013_syn_final --N 17 --use_top_env --use_top_base  --no_use_selection --is_split_over_user_num&
CUDA_VISIBLE_DEVICES=2 nohup python __main__.py --para --T_ratio 0.1 del+100.0 --delta 0.1 --alpha 0.01 --lamb 0.5 --tilde_alpha 0.25 --tilde_lamb 1.0 --Env=Kmeans --alg=SKmeans --seed=-1 --fn=../datasets/synthetic_datasets/con_data_50_0.01_5/ --sig 1013_syn_final --N 17 --use_top_env --use_top_base  --no_use_selection --is_split_over_user_num&
CUDA_VISIBLE_DEVICES=3 nohup python __main__.py --para --T_ratio 0.2 del+50.0 --delta 0.01 --alpha 0.25 --lamb 0.5 --tilde_alpha 0.1 --tilde_lamb 0.9 --Env=Kmeans --alg=SKmeans --seed=-1 --fn=../datasets/synthetic_datasets/con_data_50_0.1_5/ --sig 1013_syn_final --N 17 --use_top_env --use_top_base  --no_use_selection --is_split_over_user_num&

CUDA_VISIBLE_DEVICES=1 nohup python __main__.py --para del+10.0 --delta 0.01 --alpha 0.01 --lamb 0.5 --tilde_alpha 0.1 --tilde_lamb 0.1 --user_ts_alpha 0.001 --Env=Kmeans --alg=Kmeans_user_ts_cluster --seed=-1 --fn=../datasets/synthetic_datasets/con_data_50_0.05_1/ --sig 1013_syn_final --N 17 --use_top_env --use_top_base  --no_use_selection  --is_split_over_user_num&
CUDA_VISIBLE_DEVICES=5 nohup python __main__.py --para del+50.0 --delta 0.01 --alpha 0.01 --lamb 0.5 --tilde_alpha 0.01 --tilde_lamb 0.9 --user_ts_alpha 0.001 --Env=Kmeans --alg=Kmeans_user_ts_cluster --seed=-1 --fn=../datasets/synthetic_datasets/con_data_50_0.05_10/ --sig 1013_syn_final --N 17 --use_top_env --use_top_base  --no_use_selection --is_split_over_user_num&
CUDA_VISIBLE_DEVICES=6 nohup python __main__.py --para del+100.0 --delta 0.1 --alpha 0.01 --lamb 0.5 --tilde_alpha 0.25 --tilde_lamb 1.0 --user_ts_alpha 0.001 --Env=Kmeans --alg=Kmeans_user_ts_cluster --seed=-1 --fn=../datasets/synthetic_datasets/con_data_50_0.01_5/ --sig 1013_syn_final --N 17 --use_top_env --use_top_base  --no_use_selection --is_split_over_user_num&
CUDA_VISIBLE_DEVICES=6 nohup python __main__.py --para del+50.0 --delta 10.0 --alpha 0.25 --lamb 0.5 --tilde_alpha 0.1 --tilde_lamb 0.9 --user_ts_alpha 0.001 --Env=Kmeans --alg=Kmeans_user_ts_cluster --seed=-1 --fn=../datasets/synthetic_datasets/con_data_50_0.1_5/ --sig 1013_syn_final --N 17 --use_top_env --use_top_base  --no_use_selection --is_split_over_user_num&

