import numpy as np
import torch
from torch import mm, mv, ger
import conf
from conf import seeds_set
import time
import gzip
import random
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConUCB:
    def __init__(self, nu, d, T, args):
        # fix random seed
        self.seed = seeds_set[args.seed]
        self.args = args
        self.incoming_user = None
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        # Initial ConUCB Hyper-parameters
        self.lamb = args.lamb if args.lamb is not None else conf.conucb_para['lambda']
        self.tilde_lamb = args.tilde_lamb if args.tilde_lamb is not None else conf.conucb_para['tilde_lambda']
        self.sigma = conf.conucb_para['sigma']
        self.alpha = args.alpha if args.alpha is not None else conf.conucb_para['alpha']
        if args.tilde_alpha is not None:
            self.tilde_alpha = args.tilde_alpha
            self.cal_alpha = False
        elif 'tilde_alpha' in conf.conucb_para:
            self.tilde_alpha = conf.conucb_para['tilde_alpha']
            self.cal_alpha = False
        else:
            self.cal_alpha = True
        # Set setting parameters
        self.nu = nu
        self.d = d
        self.T = T
        # Initial item-level parameters
        self.S = torch.stack([torch.eye(d, device=device) * (1 - self.lamb) for _ in range(nu)])
        self.b = torch.stack([torch.zeros(d, device=device) for _ in range(nu)])
        self.Sinv = torch.stack([torch.eye(d, device=device) / (1 - self.lamb) for _ in range(nu)])
        self.theta = torch.stack([torch.zeros(d, device=device) for _ in range(nu)])
        self.N = np.zeros(nu)
        # Initial attr-level parameters
        self.S_tilde = torch.stack([torch.eye(d, device=device) * self.tilde_lamb for _ in range(nu)])
        self.b_tilde = torch.stack([torch.zeros(d, device=device) for _ in range(nu)])
        self.Sinv_tilde = torch.stack([torch.eye(d, device=device) / self.tilde_lamb for _ in range(nu)])
        self.theta_tilde = torch.stack([torch.zeros(d, device=device) for _ in range(nu)])
        self.N_tilde = np.zeros(nu)

        self.tmp_name = args.tmp_name
        self.args.user_dropout_rate = 0
        self.save_theta = args.save_theta
        self.save_theta_sampled = args.save_theta_sampled
        self.save_key_term_embedding = args.save_key_term_embedding

    def _update_inverse(self, i, x, y, tilde):
        if tilde:
            self.N_tilde[i] += 1
            self.Sinv_x = mv(self.Sinv_tilde[i], x)
            self.S_tilde[i] += ger(x, x)
            self.b_tilde[i] += y * x
            self.Sinv_tilde[i] -= ger(self.Sinv_x, self.Sinv_x) / (1 + (x * self.Sinv_x).sum())
            self.theta_tilde[i] = mv(self.Sinv_tilde[i], self.b_tilde[i])
        else:
            self.N[i] += 1
            self.Sinv_x = mv(self.Sinv[i], x)
            self.S[i] += ger(x, x) * self.lamb
            self.b[i] += y * x * self.lamb
            self.Sinv[i] -= ger(self.Sinv_x, self.Sinv_x) / (1 / self.lamb + (x * self.Sinv_x).sum())
        self.theta[i] = mv(self.Sinv[i], self.b[i] + (1 - self.lamb) * self.theta_tilde[i])

    def getProb(self, N_tilde, Sinv_tilde, Sinv, theta, topk=None):
        fv = theta
        if self.cal_alpha:
            self.tilde_alpha = np.sqrt(2 * (self.d * np.log(6) + np.log(2 * N_tilde / self.sigma + 1)))
            self.tilde_alpha += 2 * np.sqrt(self.tilde_lamb) * fv.norm()
        self.FM_Sinv = mm(self.X_t, Sinv)
        var1 = (self.FM_Sinv * self.X_t).sum(dim=1).sqrt()
        var2 = (mm(self.FM_Sinv, Sinv_tilde) * self.FM_Sinv).sum(dim=1).sqrt()
        pta_1 = mv(self.X_t, fv)
        pta_2 = var1 * self.lamb * self.alpha
        pta_3 = var2 * (1 - self.lamb) * self.tilde_alpha
        if topk is None:
            return torch.argmax(pta_1 + pta_2 + pta_3)
        else:
            return torch.topk(pta_1 + pta_2 + pta_3, topk).indices

    def getCredit(self, Minv, tilde_Minv):
        tilde_Minv_FM = mm(tilde_Minv, self.tildeX_t)
        norm_M = torch.chain_matmul(self.X_t, Minv, tilde_Minv_FM).norm(dim=0)
        result_b = 1 + (self.tildeX_t * tilde_Minv_FM).sum(dim=0)
        return torch.argmax(norm_M * norm_M / result_b)

    def topK(self, i, ratio=0.1):
        topk = int(len(self.X_t) * ratio)
        return self.getProb(self.N_tilde[i], self.Sinv_tilde[i], self.Sinv[i], self.theta[i], topk=topk)

    def recommend(self, i):
        return self.getProb(self.N_tilde[i], self.Sinv_tilde[i], self.Sinv[i], self.theta[i])

    def recommend_sumper_arm(self, i):
        return self.getCredit(self.Sinv[i], self.Sinv_tilde[i])

    def store_info(self, i, x, y):
        self._update_inverse(i, x, y, tilde=False)

    def store_info_suparm(self, i, x, y):
        self._update_inverse(i, x, y, tilde=True)

    def update(self, t):
        return self.nu

    def getAddiBudget(self, cur_bt, iter_):
        left_budget = cur_bt(iter_) - (0 if iter_ == 0 else cur_bt(iter_ - 1))
        return int(left_budget) if left_budget > 0 else -1

    def getAlgorithmFileName(self, prefix):
        fn = f'topk_[{self.__class__.__name__}+alpha+{self.alpha}'
        fn += f"+lamb{self.lamb}"
        fn += f"+talpha{self.tilde_alpha}"
        fn += f"+tlamb{self.tilde_lamb}"
        fn += ']'
        if not self.args.no_use_selection:
            fn += "-selection"
        else:
            fn += "-no_selection"

        fn = prefix + fn
        return fn

    def user_dropout(self, t, r=0.0):
        if t % int(self.T // 10) == 0 and r > 0:
            select = np.arange(len(self.N))
            random.shuffle(select)
            select = select[:int(len(self.N) * r)]
            # Erase item-level parameters
            self.S[select] = torch.eye(self.d, device=device) * (1 - self.lamb)
            self.b[select] = torch.zeros(self.d, device=device)
            self.Sinv[select] = torch.eye(self.d, device=device) / (1 - self.lamb)
            self.theta[select] = torch.zeros(self.d, device=device)
            self.N[select] = 0
            # Erase attr-level parameters
            self.S_tilde[select] = torch.eye(self.d, device=device) * self.tilde_lamb
            self.b_tilde[select] = torch.zeros(self.d, device=device)
            self.Sinv_tilde[select] = torch.eye(self.d, device=device) / self.tilde_lamb
            self.theta_tilde[select] = torch.zeros(self.d, device=device)
            self.N_tilde[select] = 0
            self.incoming_user = select
        elif t % int(self.T // 10) > self.nu * r * 5:
            self.incoming_user = None
        return self.incoming_user

    def save_cluster_num(self, fn, t):
        pass

    def save_user_theta(self, fn, t):
        pass

    def run(self, envir):
        # set file name
        fn = self.getAlgorithmFileName(envir.file_name)
        self.fn = fn

        with open(self.fn + "cluster_num_from_delta.csv", "w") as f:
            pass
        f_key_term_rank = open(fn + 'key_term_rank.txt', 'w')
        f = gzip.open(fn + '.gz', 'wt')
        # initial some variables
        total_reward, t = 0, 0
        self.X_t, self.tildeX_t = envir.X_t, envir.tildeX_t
        f.write(','.join(['iteration', 'average_reward', 'cluster_num', 'user_id', 'user_reward', 't']) + '\n')
        # main
        if self.save_key_term_embedding:
            for i in range(self.nu):
                os.makedirs(f'{fn}/semantic/user_{i}/', exist_ok=True)
        start = time.time()
        while t < self.T:
            self.I = envir.generate_users(self.user_dropout(t, self.args.user_dropout_rate))
            for i in self.I:
                rounds = envir.get_rounds()
                for _ in range(rounds):
                    Addi_budget = self.getAddiBudget(conf.bt, self.N[i])
                    if Addi_budget > 0:
                        if not self.args.no_use_selection:
                            try:
                                selection = self.topK(i, ratio=0.1)
                            except:
                                selection = None
                        else:
                            selection = None
                        self.tildeX_t = envir.update(self.N.mean(), selection=selection)

                    for key_term_ask in range(Addi_budget):
                        k_tilde = self.recommend_sumper_arm(i=i)

                        top_ratio = envir.rank_key_term(i, k_tilde)
                        f_key_term_rank.write(f"{t},{top_ratio}\n")
                        x_tilde, r_tilde, _ = envir.feedback(i=i, k=k_tilde, kind='attr')

                        if self.save_key_term_embedding:
                            assert torch.equal(x_tilde, envir.tildeX_t[:, k_tilde])
                            assert torch.equal(r_tilde, envir.user_attr[i, k_tilde])
                            attribute_item = {
                                'k_selected': k_tilde,
                                'fv': x_tilde.cpu().numpy().ravel().tolist(),
                                'items': envir.select[k_tilde].coalesce().indices().cpu().tolist(),
                                'reward': r_tilde.cpu().numpy().ravel().tolist(),
                            }
                            np.save(f'{fn}/semantic/user_{i}/{t}.{key_term_ask}.npy', attribute_item)

                        self.store_info_suparm(i=i, x=x_tilde, y=r_tilde)
                        self.update(t)
                    k = self.recommend(i=i)
                    x, r, reg = envir.feedback(i=i, k=k, kind='item')
                    self.store_info(i=i, x=x, y=r)
                    cluster_num = self.update(t)
                    total_reward += r
                    average_reward = total_reward / (t + 1)
                    f.write(','.join([
                        str(t), str(average_reward.tolist()), str(cluster_num),
                        str(i), str(r.tolist()), str(time.time())
                    ]) + '\n')

                    if t % 1000 == 999:
                        f.close()
                        print('======== Iter:%d, Cluster-Num:%d, Attr-Num:%d, Reward:%f, time:%f ========' % (
                            t, cluster_num, self.tildeX_t.shape[1], total_reward / (t + 1), time.time() - start))
                        print('Iteration: %d, User: %d, user Reward: %f' % (t, i, r))
                        f = gzip.open(fn + '.gz', 'at')

                        f_key_term_rank = open(fn + 'key_term_rank.txt', 'a+')
                        start = time.time()

                    if self.save_theta:
                        if (t + 1) % 50 == 0:
                            self.save_cluster_num(fn, t)
                            self.save_user_theta(fn, t)

                    t += 1


class Cluster:
    def __init__(self, users, S, b, N, S_tilde, b_tilde, N_tilde, checks=None):
        # Initial cluster Hyper-parameters
        self.lamb = conf.conucb_para['lambda']
        self.users = users  # a list/array of users
        # update cluster parameter
        self.S = S.clone().detach()
        self.b = b.clone().detach()
        self.N = N
        self.Sinv = None
        # update cluster tilde parameter
        self.S_tilde = S_tilde.clone().detach()
        self.b_tilde = b_tilde.clone().detach()
        self.N_tilde = N_tilde
        self.Sinv_tilde = None
        # update theta
        self.theta_tilde = None
        self.theta = None
        # update checks
        if checks is not None:
            self.checks = checks
            self.checked = len(self.users) == sum(self.checks.values())

    def update_check(self, i):
        self.checks[i] = True
        self.checked = len(self.users) == sum(self.checks.values())

    def update_inverse(self, x, y, tilde):
        if tilde:
            self.N_tilde += 1
            self.Sinv_x = mv(self.Sinv_tilde, x)
            self.S_tilde += ger(x, x)
            self.b_tilde += y * x
            self.Sinv_tilde -= ger(self.Sinv_x, self.Sinv_x) / (1 + (self.Sinv_x * x).sum())
            self.theta_tilde = mv(self.Sinv_tilde, self.b_tilde)
        else:
            self.N += 1
            self.Sinv_x = mv(self.Sinv, x)
            self.S += ger(x, x) * self.lamb
            self.b += y * x * self.lamb
            self.Sinv -= ger(self.Sinv_x, self.Sinv_x) / (1 / self.lamb + (self.Sinv_x * x).sum())
        self.theta = mv(self.Sinv, self.b + (1 - self.lamb) * self.theta_tilde)


class ConCLuster(ConUCB):
    def __init__(self, nu, d, T, args):
        super(ConCLuster, self).__init__(nu, d, T, args)
        self.clusters = {
            0: Cluster(
                users=list(range(nu)), N=0, N_tilde=0,
                S=torch.eye(d, device=device) * (1 - self.lamb),
                b=torch.zeros(d, device=device),
                S_tilde=torch.eye(d, device=device) * self.tilde_lamb,
                b_tilde=torch.zeros(d, device=device),
                checks={i: False for i in range(nu)}
            )
        }
        self.clusters[0].Sinv = torch.eye(d, device=device) / (1 - self.lamb)
        self.clusters[0].Sinv_tilde = torch.eye(d, device=device) / self.tilde_lamb
        self.clusters[0].theta_tilde = torch.zeros(d, device=device)
        self.clusters[0].theta = torch.zeros(d, device=device)
        self.cluster_inds = np.zeros(nu)

    def recommend(self, i):
        cluster = self.clusters[self.cluster_inds[i]]
        return self.getProb(cluster.N_tilde, cluster.Sinv_tilde, cluster.Sinv, cluster.theta)

    def recommend_sumper_arm(self, i):
        cluster = self.clusters[self.cluster_inds[i]]
        return self.getCredit(cluster.Sinv, cluster.Sinv_tilde)

    def store_info(self, i, x, y):
        super(ConCLuster, self).store_info(i, x, y)
        c = self.cluster_inds[i]
        self.clusters[c].update_inverse(x, y, tilde=False)

    def store_info_suparm(self, i, x, y):
        super(ConCLuster, self).store_info_suparm(i, x, y)
        c = self.cluster_inds[i]
        self.clusters[c].update_inverse(x, y, tilde=True)

    def save_cluster_num(self, fn, t):
        # cluster_num_each_cluster = np.zeros(len(self.clusters))
        cluster_belong_to = np.zeros(self.nu)
        for index, key in enumerate(self.clusters.keys()):
            # cluster_num_each_cluster[index] = len(self.clusters[key].users)
            for u in self.clusters[key].users:
                cluster_belong_to[u] = index

        with open(fn + "cluster_belong_to.csv", "a+") as f:
            f.write(str(t) + ',')
            f.write(','.join([
                str(int(x)) for x in cluster_belong_to
            ]) + '\n')

    def save_user_theta(self, fn, t):
        os.makedirs(fn + "_theta", exist_ok=True)
        fn_theta = fn + f"_theta/theta{t}.npy"
        np.save(fn_theta, self.theta.cpu())

    def run(self, envir):
        fn = self.getAlgorithmFileName(envir.file_name)
        open(fn + "cluster_belong_to.csv", "w")
        super(ConCLuster, self).run(envir)
