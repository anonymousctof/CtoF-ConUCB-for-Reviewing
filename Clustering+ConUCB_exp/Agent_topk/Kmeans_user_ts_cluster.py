import numpy as np
import torch
from Agent_topk.Kmeans import Kmeans

from Env.utlis_bk import KMeans, evolve, evolve_2, evolve_3
import time
import copy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
class Kmeans_user_ts_cluster(Kmeans):
    def __init__(self, nu, d, T, args):
        self.user_ts_alpha = args.user_ts_alpha
        self.user_cluster_type = args.user_cluster_type
        self.cluster_lower = args.cluster_lower
        self.is_split_over_user_num = args.is_split_over_user_num
        super(Kmeans_user_ts_cluster, self).__init__(nu, d, T, args)

    def getAlgorithmFileName(self, prefix):
        fn = 'topk_[' + self.__class__.__name__
        if self.init_num is None:
            fn += '+' + '+'.join(['Dyn', str(self._delta)])
        else:
            fn += '+' + '+'.join(['Fix', str(self.init_num)])
        fn += '+' + '+'.join(["kalpha", str(self.alpha), "ktalpha", str(self.tilde_alpha)])
        fn += f"+lamb{self.lamb}+tlamb{self.tilde_lamb}+user_ts_a{self.user_ts_alpha}"
        if self.user_cluster_type is not None:
            fn += f'+cluster_t+{self.user_cluster_type}'
            pass
        if self.cluster_lower is not None:
            fn += f"+lower+{self.cluster_lower}"

        if self.is_split_over_user_num:
            fn += f"+is_split_over_user_num"
        fn += ']'

        fn = prefix + fn
        return fn
    def Run_Kmeans(self, t):
        if self.init_num is None:
            if self.user_cluster_type == "2":
                n_cluster = evolve_2(self.T, t, len(self.N), self._delta, self.cluster_lower)
            elif self.user_cluster_type == "3":
                n_cluster = evolve_3(self.T, t, len(self.N), self._delta, self.cluster_lower)

            else:
                n_cluster = evolve(self.T, t, len(self.N), self._delta)
        else:
            n_cluster = self.init_num

        assert self.theta.shape == (self.nu, self.d)
        assert self.Sinv.shape == (self.nu, self.d, self.d)
        start_time = time.time()
        if self.user_ts_alpha == 0:
            theta_ts_sampled = copy.deepcopy(self.theta)
        else:
            theta_ts_sampled = torch.distributions.MultivariateNormal(self.theta, self.user_ts_alpha * self.Sinv).sample()
        self.X_dist = torch.cdist(theta_ts_sampled, theta_ts_sampled, p=2)
        assert theta_ts_sampled.shape == (self.nu, self.d)
        assert self.X_dist.shape == (self.nu, self.nu)

        cluster_inds, select, weights = KMeans(
            n_clusters=n_cluster, random_state=self.seed, max_iter=self.max_iter,
            select=self.I, X_dist=self.X_dist
        ).fit_predict(theta_ts_sampled)
        # ).fit_predict(self.theta)

        kmeans_cluster_num = select.shape[0]
        if self.is_split_over_user_num == True:
            if n_cluster >= self.nu:
                if select.shape[0] < self.nu:
                    cluster_inds = np.arange(self.nu)
                    x = torch.arange(self.nu, device=device)
                    ones = torch.ones_like(x).float()
                    select = torch.sparse_coo_tensor(torch.stack([x, x]), ones, (self.nu, self.nu))
                    weights = torch.sparse.sum(select, dim=1).to_dense()


        with open(self.fn + "cluster_num_from_delta.csv", "a+") as f:
            f.write(str(t) + ',')
            f.write(f'{t},{kmeans_cluster_num},{select.shape[0]},{n_cluster}\n')
        self.user_information_aggregate(cluster_inds, select, weights)
        if self.save_theta_sampled:
            if (t + 1) % 5000 == 0:
                self.save_user_theta_sampled(self.fn, t, theta_ts_sampled)

    def update(self, t):
        if t % 50 == 49:
            self.Run_Kmeans(t)
        return len(self.clusters)

    def save_user_theta_sampled(self, fn, t, theta_sampled):
        os.makedirs(fn + "_theta_sampled", exist_ok=True)
        fn_theta = fn + f"_theta_sampled/theta_sampled{t}.npy"
        np.save(fn_theta, theta_sampled.cpu())