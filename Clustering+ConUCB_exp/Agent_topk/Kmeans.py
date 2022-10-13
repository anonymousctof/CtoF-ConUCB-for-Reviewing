import numpy as np
import torch
from Agent_topk.BASE import ConCLuster, Cluster
import conf
import random
from conf import seeds_set
from Env.utlis_bk import KMeans, evolve, evolve_2, evolve_3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Kmeans(ConCLuster):
    def __init__(self, nu, d, T, args):
        super(Kmeans, self).__init__(nu, d, T, args)
        # fix random seed
        self.seed = seeds_set[args.seed]
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        # Initial Kmeans Hyper-parameters
        self._delta = args.delta if args.delta is not None else conf.kmeans_para['delta']
        self.max_iter = args.max_iter if args.max_iter is not None else conf.kmeans_para['max_iter']
        self.init_num = args.cluster_num
        self.X_dist = torch.zeros(nu, nu, device=device)

        self.user_cluster_type = args.user_cluster_type
        self.cluster_lower = args.cluster_lower

        self.is_split_over_user_num = args.is_split_over_user_num

    def getAlgorithmFileName(self, prefix):
        fn = 'topk_[' + self.__class__.__name__
        if self.init_num is None:
            fn += '+' + '+'.join(['Dyn', str(self._delta)])
        else:
            fn += '+' + '+'.join(['Fix', str(self.init_num)])
        fn += f"+lamb{self.lamb}"
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

    def user_information_aggregate(self, cluster_inds, select, weights):
        clusters = {}
        n_clusters = len(weights)
        n_samples = self.nu

        eyes = (1 - weights.view(-1, 1, 1)) * torch.eye(self.d, device=device).expand(n_clusters, self.d, self.d)
        eyes = torch.cat([eyes * (1 - self.lamb), eyes * self.tilde_lamb], dim=0)

        S = torch.cat([self.S.reshape(n_samples, -1), self.S_tilde.reshape(n_samples, -1)], dim=1)
        S = torch.sparse.mm(select, S).reshape(n_clusters, 2, self.d, self.d
                                               ).transpose(0, 1).reshape(-1, self.d, self.d) + eyes
        torch.cuda.synchronize()
        Sinv = torch.inverse(S)

        b = torch.sparse.mm(select, torch.cat([self.b, self.b_tilde], dim=1))
        b = b.reshape(n_clusters, 2, self.d).transpose(0, 1).reshape(-1, self.d)

        theta = torch.bmm(Sinv, b.unsqueeze(2)).squeeze()
        theta[:n_clusters] += (1 - self.lamb) * torch.bmm(Sinv[:n_clusters], theta[n_clusters:].unsqueeze(2)).squeeze()

        N = torch.sparse.mm(select, torch.tensor([self.N.tolist()], device=device).T).cpu().numpy().ravel()
        N_tilde = torch.sparse.mm(select, torch.tensor([self.N_tilde.tolist()], device=device).T).cpu().numpy().ravel()

        for i in range(n_clusters):
            clusters[i] = Cluster(
                users=np.where(cluster_inds == i)[0],
                S=S[i], b=b[i], N=N[i],
                S_tilde=S[i + n_clusters], b_tilde=b[i + n_clusters], N_tilde=N_tilde[i],
            )
            clusters[i].Sinv = Sinv[i]
            clusters[i].Sinv_tilde = Sinv[i + n_clusters]
            clusters[i].theta = theta[i]
            clusters[i].theta_tilde = theta[i + n_clusters]

        self.cluster_inds = cluster_inds
        self.clusters = clusters

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

        with open(self.fn + "cluster_num_from_delta.csv", "a+") as f:
            f.write(str(t) + ',')
            f.write(f'{n_cluster}\n')

        cluster_inds, select, weights = KMeans(
            n_clusters=n_cluster, random_state=self.seed, max_iter=self.max_iter,
            select=self.I, X_dist=self.X_dist
        ).fit_predict(self.theta)

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

    def update(self, t):
        dist = (self.theta[self.I] - self.theta).pow(2).sum(dim=1).sqrt()
        self.X_dist[self.I[0]] = dist
        self.X_dist[:, self.I[0]] = dist
        if t % 50 == 49:
            self.Run_Kmeans(t)
        return len(self.clusters)
