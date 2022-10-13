
screen -S missing_key_delete
conda activate pytorch14
cd /mnt/qzli_hdd/ConUCB/cluster_original/Clustering+ConUCB_exp
cd /root/data/qzli/ConUCB/cluster_original/Clustering+ConUCB_exp/

CUDA_VISIBLE_DEVICES=5 nohup python __main__.py --alpha 0.01 --tilde_alpha 0.25 --lamb 0.5 --tilde_lamb 0.9 --Env=Attr --alg=ConUCB --seed=-1 --fn=../dataset-generate/data3_movielens_generator/ml_25m_0401a1_un2000_in25000_0 --sig 1006_n1_final --N 16 --use_top_env --use_top_base   --item_rate 0.4  --delete_key_term high&
CUDA_VISIBLE_DEVICES=5 nohup python __main__.py --alpha 0.01 --tilde_alpha 0.25 --lamb 0.5 --tilde_lamb 0.9 --Env=Attr --alg=ConUCB --seed=-1 --fn=../dataset-generate/data3_movielens_generator/ml_25m_0401a1_un2000_in25000_0 --sig 1006_n1_final --N 16 --use_top_env --use_top_base   --item_rate 0.6  --delete_key_term high&

CUDA_VISIBLE_DEVICES=1 nohup python __main__.py --alpha 1 --tilde_alpha 0.25 --lamb 0.5 --tilde_lamb 0.1 --Env=Attr --alg=ConUCB --seed=-1 --fn=../dataset-generate/data4_lastfm/un_1500-bn_10000/train_item_user_test_rate --sig 1006_n1_final --N 16 --use_top_env --use_top_base   --item_rate 0.4  --delete_key_term high&
CUDA_VISIBLE_DEVICES=1 nohup python __main__.py --alpha 1 --tilde_alpha 0.25 --lamb 0.5 --tilde_lamb 0.1 --Env=Attr --alg=ConUCB --seed=-1 --fn=../dataset-generate/data4_lastfm/un_1500-bn_10000/train_item_user_test_rate --sig 1006_n1_final --N 16 --use_top_env --use_top_base   --item_rate 0.6  --delete_key_term high&

