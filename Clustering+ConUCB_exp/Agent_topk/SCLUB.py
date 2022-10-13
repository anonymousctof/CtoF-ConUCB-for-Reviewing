import numpy as np
import random
import torch
from torch import mv
from Agent_topk.BASE import ConCLuster, Cluster
import conf
from conf import seeds_set
from Env.utlis import factT, is_power2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SCLUB(ConCLuster):
    def __init__(self, nu, d, T, args):
        super(SCLUB, self).__init__(nu, d, T, args)
        # fix random seed
        self.seed = seeds_set[args.seed]
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        # Initial SCLUB Hyper-parameters
        self._alpha = args.sclub_alpha if args.sclub_alpha is not None else conf.sclub_para['alpha']
        self._alpha_p = args.alpha_p if args.alpha_p is not None else conf.sclub_para['alpha_p']

    def getAlgorithmFileName(self, prefix):
        fn = 'topk_[' + self.__class__.__name__
        fn += '+' + '+'.join(['alpha', str(self._alpha), 'alpha_p', str(self._alpha_p)])
        fn += f"+lamb{self.lamb}"
        fn += ']'
        if self.no_use_conversation:
            fn = f'no_conversation+' + fn

        fn = prefix + fn
        return fn

    # alpha = 2 * np.sqrt(2 * self.d)
    def _split_or_merge(self, theta, N1, N2):
        return theta.norm() > self._alpha * (factT(N1) + factT(N2))

    def _cluster_avg_freq(self, c, t):
        return self.clusters[c].N / (len(self.clusters[c].users) * t)

    def _split_or_merge_p(self, p1, p2, t):
        return np.abs(p1 - p2) > self._alpha_p * factT(t)

    def _find_available_index(self):
        cmax = max(self.clusters) + 1
        remain = set(range(cmax)) - set(self.clusters)
        return remain.pop() if len(remain) > 0 else cmax

    def _split(self, i, t):
        c = self.cluster_inds[i]
        cluster = self.clusters[c]
        cluster.update_check(i)
        if len(cluster.users) < 2:
            return

        self.cluster_thetas = torch.stack([self.theta[i], cluster.theta])
        assert not torch.isnan(self.cluster_thetas).any(), print(self.cluster_thetas)

        if (
                self._split_or_merge_p(self.N[i] / (t + 1), self._cluster_avg_freq(c, t + 1), t + 1)
                or self._split_or_merge(self.cluster_thetas[0] - self.cluster_thetas[1], self.N[i], cluster.N)
        ):
            c_new = self._find_available_index()
            self.clusters[c_new] = Cluster(
                users=[i], N=self.N[i], N_tilde=self.N_tilde[i],
                S=self.S[i],
                b=self.b[i],
                S_tilde=self.S_tilde[i],
                b_tilde=self.b_tilde[i],
                checks={i: True}
            )
            self.clusters[c_new].Sinv = self.Sinv[i].clone().detach()
            self.clusters[c_new].Sinv_tilde = self.Sinv_tilde[i].clone().detach()
            self.clusters[c_new].theta_tilde = mv(self.clusters[c_new].Sinv_tilde, self.clusters[c_new].b_tilde)
            self.clusters[c_new].theta = mv(
                self.clusters[c_new].Sinv,
                self.clusters[c_new].b + (1 - self.clusters[c_new].lamb) * self.clusters[c_new].theta_tilde
            )
            assert not torch.isnan(self.clusters[c_new].theta_tilde).any(), print(self.clusters[c_new].theta_tilde)

            assert not torch.isnan(self.clusters[c_new].theta).any(), print(self.clusters[c_new].theta)

            self.cluster_inds[i] = c_new

            cluster.users.remove(i)
            cluster.checks.pop(i)
            cluster.checked = len(cluster.users) == sum(cluster.checks.values())
            # update cluster parameter
            cluster.S -= self.S[i] - torch.eye(self.d, device=device) * (1 - self.lamb)
            cluster.b -= self.b[i]
            cluster.N -= self.N[i]
            # update cluster tilde parameter
            cluster.S_tilde -= self.S_tilde[i] - torch.eye(self.d, device=device) * self.tilde_lamb
            cluster.b_tilde -= self.b_tilde[i]
            cluster.N_tilde -= self.N_tilde[i]
            # update cluster inverse
            inverse_tmp = torch.inverse(torch.stack([cluster.S, cluster.S_tilde]))
            cluster.Sinv = inverse_tmp[0]
            cluster.Sinv_tilde = inverse_tmp[1]
            # update theta
            cluster.theta_tilde = torch.mv(cluster.Sinv_tilde, cluster.b_tilde)
            cluster.theta = torch.mv(cluster.Sinv, cluster.b + (1 - self.lamb) * cluster.theta_tilde)
            assert not torch.isnan(cluster.theta).any(), print("cluster.theta", cluster.theta,
                                                               "cluster.Sinv", cluster.Sinv,
                                                               "cluster.b", cluster.b,
                                                               "cluster.theta_tilde", cluster.theta_tilde,
                                                               "inverse_tmp", inverse_tmp,
                                                               "cluster.S", cluster.S,
                                                               "torch.isnan(cluster.S).any()", torch.isnan(cluster.S).any(),
                                                               "cluster.S_tilde", cluster.S_tilde,)

        if len(cluster.users) == 0:
            del self.clusters[c]

    def _merge(self, t):
        c_list = set([i if self.clusters[i].checked else None for i in self.clusters]) - {None}
        c_list = list(c_list)
        if len(c_list) < 2:
            return

        def _factT(T):
            return torch.sqrt((1 + torch.log(1 + T)) / (1 + T))

        self.cluster_thetas = torch.stack([self.clusters[c].theta for c in c_list])
        assert not torch.isnan(self.cluster_thetas).any(), print(self.cluster_thetas)

        self.cluster_Ns = _factT(torch.tensor([self.clusters[c].N for c in c_list], device=device)).unsqueeze(1)
        self.judgements = torch.cdist(self.cluster_thetas, self.cluster_thetas) - \
                          self._alpha * (self.cluster_Ns + self.cluster_Ns.T) / 2

        new_clusters = dict()
        labels = dict()
        for [i, j] in (self.judgements.triu(diagonal=1) < 0).nonzero(as_tuple=False):
            c1, c2 = c_list[i], c_list[j]
            if c1 not in labels and c2 not in labels:
                labels[c1] = c1
                labels[c2] = c1
                new_clusters[c1] = {c2}
            elif c1 in labels and c2 not in labels:
                labels[c2] = labels[c1]
                new_clusters[labels[c1]].update({c2})
            elif c1 not in labels and c2 in labels:
                labels[c1] = labels[c2]
                new_clusters[labels[c2]].update({c1})

        if len(new_clusters) == 0:
            return

        for c in new_clusters:
            # update cluster parameter
            self.clusters[c].S += torch.stack([self.clusters[c2].S for c2 in new_clusters[c]]).sum(dim=0) - \
                                  torch.eye(self.d, device=device) * (1 - self.lamb) * len(new_clusters[c])
            self.clusters[c].b += torch.stack([self.clusters[c2].b for c2 in new_clusters[c]]).sum(dim=0)
            self.clusters[c].N += np.asarray([self.clusters[c2].N for c2 in new_clusters[c]]).sum()
            # update cluster tilde parameter
            self.clusters[c].S_tilde += torch.stack([self.clusters[c2].S_tilde for c2 in new_clusters[c]]).sum(dim=0) - \
                                        torch.eye(self.d, device=device) * self.tilde_lamb * len(new_clusters[c])
            self.clusters[c].b_tilde += torch.stack([self.clusters[c2].b_tilde for c2 in new_clusters[c]]).sum(dim=0)
            self.clusters[c].N_tilde += np.asarray([self.clusters[c2].N_tilde for c2 in new_clusters[c]]).sum()
            for c2 in new_clusters[c]:
                self.clusters[c].users.extend(self.clusters[c2].users)
                self.clusters[c].checks.update(self.clusters[c2].checks)
                self.cluster_inds[self.clusters[c2].users] = c
                self.clusters.pop(c2)
            self.clusters[c].checked = len(self.clusters[c].users) == sum(self.clusters[c].checks.values())

        batch_inverse = torch.inverse(torch.cat([
            torch.stack([self.clusters[c].S for c in new_clusters.keys()]),
            torch.stack([self.clusters[c].S_tilde for c in new_clusters.keys()])
        ], dim=0))
        for index, c in enumerate(new_clusters.keys()):
            self.clusters[c].Sinv = batch_inverse[index]
            self.clusters[c].Sinv_tilde = batch_inverse[index + len(new_clusters)]
            self.clusters[c].theta_tilde = torch.mv(self.clusters[c].Sinv_tilde, self.clusters[c].b_tilde)
            self.clusters[c].theta = torch.mv(
                self.clusters[c].Sinv,
                self.clusters[c].b + (1 - self.clusters[c].lamb) * self.clusters[c].theta_tilde
            )
            assert not torch.isnan(self.clusters[c].theta).any(), print("self.clusters[c].theta", self.cluster[c].theta,
                                                                        "self.clusters[c].S", self.clusters[c].S,
                                                                        "self.clusters[c].S_tilde",
                                                                        self.clusters[c].S_tilde,
                                                                        "self.clusters[c].Sinv", self.clusters[c].Sinv,
                                                                        "self.clusters[c].b", self.clusters[c].b,
                                                                        "self.clusters[c].b_tilde",
                                                                        self.clusters[c].b_tilde)

    def update(self, t):
        for i in self.I:
            self._split(i, t)
        if t % len(self.N) == len(self.N) - 1:
            self._merge(t)
        if is_power2(t):
            for key in self.clusters:
                self.clusters[key].checks = {i: False for i in self.clusters[key].users}
                self.clusters[key].checked = False
        return len(self.clusters)
