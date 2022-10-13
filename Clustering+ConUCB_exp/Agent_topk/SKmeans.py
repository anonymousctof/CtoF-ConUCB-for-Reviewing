import numpy as np
import torch
from Agent_topk.SCLUB import SCLUB
from Agent_topk.Kmeans import Kmeans
import conf
import random
from conf import seeds_set

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SKmeans(SCLUB, Kmeans):
    def __init__(self, nu, d, T, args):
        SCLUB.__init__(self, nu, d, T, args)
        # fix random seed
        self.seed = seeds_set[args.seed]
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        # Initial SKmeans Hyper-parameters
        self._delta = args.delta if args.delta is not None else conf.kmeans_para['delta']
        self.max_iter = args.max_iter if args.max_iter is not None else conf.kmeans_para['max_iter']
        self.init_num = args.cluster_num
        self.sclub_alpha = args.sclub_alpha
        self.T_ratio = args.T_ratio
        self.is_split_over_user_num = args.is_split_over_user_num

    def getAlgorithmFileName(self, prefix):
        fn = 'topk_[' + self.__class__.__name__
        if self.init_num is None:
            fn += '+' + '+'.join(['Dyn', str(self._delta)])
        else:
            fn += '+' + '+'.join(['Fix', str(self.init_num)])
        fn += '+' + '+'.join(['salpha', str(self._alpha), 'salpha_p', str(self._alpha_p), "kalpha", str(self.alpha), "ktalpha", str(self.tilde_alpha)])
        fn += f"+lamb{self.lamb}"
        fn += f"+tlamb{self.tilde_lamb}"
        if self.user_cluster_type is not None:
            fn += f'+cluster_t+{self.user_cluster_type}'
            pass
        if self.cluster_lower is not None:
            fn += f"+lower+{self.cluster_lower}"
        fn += f"+T_ratio{self.T_ratio}"
        if self.is_split_over_user_num:
            fn += f"+is_split_over_user_num"
        fn += ']'
        fn = f"+T_ratio{self.T_ratio}" + fn


        fn = prefix + fn
        return fn

    def update(self, t):
        if t < int(self.T * self.T_ratio):
            SCLUB.update(self, t)
        else:
            Kmeans.update(self, t)
        return len(self.clusters)
