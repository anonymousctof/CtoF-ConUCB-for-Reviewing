import os
import time
import argparse

from Env.ENVIRONMENT import Environment
from Env.ENVIRONMENT_top import Environment as Environment_top

# python

def main(args, filename):
    if args.use_top_env:
        envir = Environment_top(args=args, file_name=filename, seed=args.seed)
    else:
        envir = Environment(args=args, file_name=filename, seed=args.seed)
    d, num_users = envir.d, envir.nu

    if args.use_top_base:
        if args.alg == 'ConUCB':
            from Agent_topk.BASE import ConUCB
            agent = ConUCB(nu=num_users, d=d, T=2 ** args.N - 1, args=args)
        elif args.alg == 'LinUCB':
            from Agent_topk.LinUCB import LinUCB
            agent = LinUCB(nu=num_users, d=d, T=2 ** args.N - 1, args=args)
        elif args.alg == 'ArmCon':
            from Agent_topk.ArmCon import ArmCon
            agent = ArmCon(nu=num_users, d=d, T=2 ** args.N - 1, args=args)
        elif args.alg == 'Kmeans':
            from Agent_topk.Kmeans import Kmeans
            agent = Kmeans(nu=num_users, d=d, T=2 ** args.N - 1, args=args)
        elif args.alg == 'SKmeans':
            from Agent_topk.SKmeans import SKmeans
            agent = SKmeans(nu=num_users, d=d, T=2 ** args.N - 1, args=args)
        elif args.alg == 'Kmeans_user_ts_cluster':
            from Agent_topk.Kmeans_user_ts_cluster import Kmeans_user_ts_cluster
            agent = Kmeans_user_ts_cluster(nu=num_users, d=d, T=2 ** args.N - 1, args=args)
        else:
            raise AssertionError
    else:
        raise AssertionError
    agent.run(envir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', dest='seed', default=9, type=int)
    parser.add_argument('--N', dest='N', default=17, type=int)
    parser.add_argument('--fn', dest='fn', default='../dataset-generate/data5_lastfm/lastfm', type=str)
    parser.add_argument('--Env', dest='Env', default='Kmeans_5', type=str)
    parser.add_argument('--para', dest='para', default=None, type=str)
    parser.add_argument('--alg', dest='alg', default='ConUCB', type=str)
    parser.add_argument('--alpha', dest='alpha', default=None, type=float)
    parser.add_argument('--sclub_alpha', type=float, default=None)
    parser.add_argument('--alpha_p', dest='alpha_p', default=None, type=float)
    parser.add_argument('--delta', dest='delta', default=None, type=float)
    parser.add_argument('--cluster_num', dest='cluster_num', default=None, type=int)
    parser.add_argument('--max_iter', dest='max_iter', default=None, type=int)
    parser.add_argument('--sig', dest='sig',  default="", type=str)
    parser.add_argument('--lamb', default=None, type=float)
    parser.add_argument('--tilde_lamb', default=None, type=float)
    parser.add_argument("--no_use_con", action="store_true")
    parser.add_argument("--tmp_name", default="", type=str)
    parser.add_argument("--use_linucb_rec", action="store_true")
    parser.add_argument("--use_linucb_update", action="store_true")
    parser.add_argument("--use_torch_inverse", action="store_true")
    parser.add_argument("--use_1_inverse", action="store_true")
    parser.add_argument("--use_lamb_inverse", action="store_true")
    parser.add_argument("--tilde_alpha", default=None, type=float)
    parser.add_argument("--user_ts_alpha", default=None, type=float)

    parser.add_argument('--use_theta_sampled_cluster', action='store_true')
    parser.add_argument('--no_use_conversation', action='store_true')
    parser.add_argument('--recommend_type', type=str, default=None)
    parser.add_argument('--T_ratio', type=float, default=0.1)
    # self.user_cluster_type, self.cluster_lower
    parser.add_argument("--user_cluster_type", default=None, type=str)
    parser.add_argument("--cluster_lower", default=None, type=float)
    parser.add_argument("--use_top_env", action='store_true')
    parser.add_argument("--use_top_base", action='store_true')
    parser.add_argument("--no_use_selection", action='store_true')

    parser.add_argument('--item_rate', dest='item_rate', default=1.0, type=float)
    parser.add_argument('--delete_key_term', default="high", type=str)
    parser.add_argument('--use_seed_folder', action='store_true')

    parser.add_argument('--save_theta', action='store_true')
    parser.add_argument('--save_theta_sampled', action='store_true')
    parser.add_argument('--save_key_term_embedding', action='store_true')

    parser.add_argument('--is_split_over_user_num', action='store_true')

    args = parser.parse_args()
    prefix = args.fn

    start_time = time.time()
    if args.seed < 0:
        for i in range(abs(args.seed) - 1, 10):
            args.seed = i
            # args.fn = prefix + '/' + str(i)
            args.fn = prefix + '/' + args.sig + '/' + str(i)
            # if not os.path.exists(args.fn):
            #     os.mkdir(args.fn)
            os.makedirs(args.fn, exist_ok=True)
            main(args=args, filename=prefix)
    else:
        if args.use_seed_folder:
            args.fn = prefix + '/' + args.sig + '/' + str(args.seed)
        else:
            args.fn = prefix + '/' + args.sig + '/'
        os.makedirs(args.fn, exist_ok=True)
        main(args=args, filename=prefix)
    run_time = time.time() - start_time
    print('total time:', run_time)
