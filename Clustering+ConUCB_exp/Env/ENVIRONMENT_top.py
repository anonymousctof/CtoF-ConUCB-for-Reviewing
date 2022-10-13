import numpy as np
import json
import torch
from Env import Arm
from conf import armNoiseScale, suparmNoiseScale, minRecommend, maxRecommend, seeds_set
import random
from Env.utlis_bk import KMeans, evolve
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Environment:
    # p: frequency vector of users
    def __init__(self, args, file_name, seed):
        self.delete_key_term = args.delete_key_term

        self.file_name = file_name
        self.args = args
        # fix random seed
        self.seed = seeds_set[seed]
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        print("Seed = ", self.seed)
        # interaction frequency
        self.min_rounds = minRecommend
        self.max_rounds = maxRecommend
        print('# Interaction frequency:', minRecommend, '-', maxRecommend)
        # load User
        # load Arm
        self.AM = Arm.ArmManager(file_name)
        self.AM.loadArms()
        self.X_t = torch.stack([v.fv for _, v in self.AM.arms.items()])
        print('# Arms loaded:', self.X_t.shape[0])
        # load Review
        self.user_item, self.user_attr = self.load_reviews(file_name), None
        self.nu = self.user_item.shape[0]
        self.d = self.X_t.shape[1]
        # load Attribute
        self.tildeX_t = self.update(0)
        print('# Superarm loaded:', self.tildeX_t.shape[1])
        # load User Distribution
        self.p = self._get_half_frequency_vector()[0]

        # set file path
        if self.args.Env == 'Attr':
            self.file_name = self.args.fn + '/[Item+' + str(self.args.item_rate) + ']+'
        else:
            self.file_name = self.args.fn + '/[' + self.args.Env + '+' + self.args.para + ']+'
        if self.delete_key_term is not None:
            self.file_name = self.file_name + f"del_k_{self.delete_key_term}+"
    def rank_key_term(self, user, key_term_index):
        rank = torch.argsort(self.user_attr[user])  # 从高往低
        key_term_rank = rank[key_term_index] * 1.0 / self.user_attr[user].shape[0]
        return key_term_rank

    def update(self, t, selection=None):
        if t > 0 and (self.args.Env == 'Attr' or self.args.para.startswith('Fix')):
            return self.tildeX_t

        if self.args.Env == 'Attr':
            user_attr, suparms = self._read_attributes()
            self.user_attr = torch.stack(user_attr).T
            self.tildeX_t = torch.stack(suparms).T
        else:
            if t == 0:
                self.X_dist = torch.cdist(self.X_t, self.X_t)
                self.n_cluster = 1
                if self.args.Env.startswith('Attr'):
                    user_attr, suparms = self._read_attributes()
                    self.user_attr_init = torch.stack(user_attr).T
                    self.tildeX_t_init = torch.stack(suparms).T

            if self.args.para.startswith('Fix'):
                n_cluster = int(self.args.para[4:])
            else:
                n_cluster = evolve(2 ** self.args.N / self.nu, t, self.X_t.shape[0], float(self.args.para[4:]))

            if self.n_cluster != n_cluster:
                self.n_cluster = n_cluster
            elif t > 0:
                return self.tildeX_t

            if self.args.Env.endswith('Kmeans'):
                if selection is not None:
                    X_t = torch.index_select(self.X_t, 0, selection)
                    user_item = torch.index_select(self.user_item, 1, selection)
                    X_dist = torch.cdist(X_t, X_t)
                else:
                    X_t, X_dist, user_item = self.X_t, self.X_dist, self.user_item
                _, select, weights = KMeans(
                    n_clusters=n_cluster, random_state=self.seed, X_dist=X_dist
                ).fit_predict(X_t)
                self.user_attr = torch.sparse.mm(select, user_item.T).T / weights
                self.tildeX_t = torch.sparse.mm(select, X_t).T / weights
                self.weights = weights
                self.select = select


            elif self.args.Env.startswith('Kmeans+'):
                gamma = float(self.args.Env.split('+')[-1])
                _, select, weights = KMeans(
                    n_clusters=n_cluster, random_state=self.seed, X_dist=self.X_dist
                ).fit_predict(self.X_t, gamma=gamma)
                self.user_attr = torch.sparse.mm(select, self.user_item.T).T / weights
                self.tildeX_t = torch.sparse.mm(select, self.X_t).T / weights

            elif self.args.Env.endswith('Kmeans_Take'):
                selects = []
                probs = []
                weights = []
                for i in range(5):
                    X_t = self.X_t[:, i * 10:i * 10 + 10]
                    _, select, weight = KMeans(
                        n_clusters=n_cluster // 5 + 1, random_state=self.seed
                    ).fit_predict(X_t)
                    probs.append(select.coalesce().values())
                    select = select.coalesce().indices()
                    select[0] = select[0] + (n_cluster // 5 + 1) * i
                    selects.append(select)
                    weights.append(weight)
                selects = torch.cat(selects, dim=1)
                probs = torch.cat(probs, dim=0)
                weights = torch.cat(weights, dim=0)
                selects = torch.sparse_coo_tensor(selects, probs, (weights.shape[0], self.X_t.shape[0]))
                self.user_attr = torch.sparse.mm(selects, self.user_item.T).T / weights
                self.tildeX_t = torch.sparse.mm(selects, self.X_t).T / weights

            elif self.args.Env.endswith('Kmeans_Leave'):
                selects = []
                probs = []
                weights = []
                for i in range(5):
                    if i == 0:
                        X_t = self.X_t[:, 10:]
                    elif i == 4:
                        X_t = self.X_t[:, :-10]
                    else:
                        X_t = torch.cat([self.X_t[:, :i * 10], self.X_t[:, i * 10 + 10:]], dim=1)
                    _, select, weight = KMeans(
                        n_clusters=n_cluster // 5 + 1, random_state=self.seed
                    ).fit_predict(X_t)
                    probs.append(select.coalesce().values())
                    select = select.coalesce().indices()
                    select[0] = select[0] + (n_cluster // 5 + 1) * i
                    selects.append(select)
                    weights.append(weight)
                selects = torch.cat(selects, dim=1)
                probs = torch.cat(probs, dim=0)
                weights = torch.cat(weights, dim=0)
                selects = torch.sparse_coo_tensor(selects, probs, (weights.shape[0], self.X_t.shape[0]))
                self.user_attr = torch.sparse.mm(selects, self.user_item.T).T / weights
                self.tildeX_t = torch.sparse.mm(selects, self.X_t).T / weights

            elif self.args.Env.endswith('GMM'):
                _, select, weights = KMeans(
                    n_clusters=n_cluster, random_state=self.seed, X_dist=self.X_dist
                ).fit_predict(self.X_t)
                from Env.utlis import GaussianMixture
                select, weights = GaussianMixture(n_components=n_cluster).fit_predict(self.X_t, select, weights)
                self.user_attr = torch.sparse.mm(select, self.user_item.T).T / weights
                self.tildeX_t = torch.sparse.mm(select, self.X_t).T / weights

            elif self.args.Env.endswith('GMM1'):
                _, select, weights = KMeans(
                    n_clusters=n_cluster, random_state=self.seed, X_dist=self.X_dist
                ).fit_predict(self.X_t)
                from Env.utlis import GaussianMixture
                select, weights = GaussianMixture(n_components=n_cluster).fit_predict(self.X_t, select, weights, top=1)
                self.user_attr = torch.sparse.mm(select, self.user_item.T).T / weights
                self.tildeX_t = torch.sparse.mm(select, self.X_t).T / weights

            else:
                print(self.args.Env)
                raise AssertionError

            if self.args.Env.startswith('Attr'):
                self.user_attr = torch.cat([self.user_attr, self.user_attr_init], dim=1)
                self.tildeX_t = torch.cat([self.tildeX_t, self.tildeX_t_init], dim=1)
        return self.tildeX_t

    def load_reviews(self, file_name):
        fn = file_name + '/user_item.npy'
        print(fn)
        if os.path.exists(fn):
            # load user_item matrix
            user_item = np.load(fn)
            user_item = torch.from_numpy(user_item).to(device).float()
            print('load user_item matrix as rewards')
            self.noise = False
        else:
            self.theta, _, _ = self.load_users()

            user_item = self.theta @ self.X_t.T
            print('generate rewards')
            self.noise = True
        return user_item

    def load_users(self):
        theta = []
        with open(self.file_name + '/user_preference.txt', 'r') as fr:
            for line in fr:
                j_s = json.loads(line)
                theta_u = j_s['preference_v']
                theta.append(theta_u)
        theta = torch.tensor(theta, device=device).squeeze()
        nu, d = theta.shape[0], theta.shape[1]
        return theta, nu, d

    def get_rounds(self):
        return np.random.randint(self.min_rounds, self.max_rounds)

    def feedback(self, i, k, kind='item'):
        if kind == 'item':
            x = torch.index_select(self.X_t, 0, k).squeeze()
            r = torch.index_select(self.user_item, 1, k)[i]
            # reg = torch.max(self.user_item[i]) - r
            reg = None
        else:
            x = torch.index_select(self.tildeX_t, 1, k).squeeze()
            r = torch.index_select(self.user_attr, 1, k)[i]
            reg = None
        if self.noise:
            r += torch.randn(1, device=device) * (armNoiseScale if kind == 'item' else suparmNoiseScale)
        return x, r.squeeze(), reg

    def generate_users(self, incoming_user=None ):
        if incoming_user is None:
            I = np.random.multinomial(1, self.p)
            X = np.nonzero(I)[0]
        else:
            X = np.random.choice(incoming_user, (1,))
        return X

    def _get_half_frequency_vector(self, m=10):
        p0 = list(np.random.dirichlet(np.ones(m)))
        p = np.ones(self.nu)
        k = int(self.nu / m)
        for j in range(m):
            for i in range(k * j, k * (j + 1)):
                p[i] = p0[j] / k
        ps = [list(np.ones(self.nu) / self.nu), list(p), list(np.random.dirichlet(np.ones(self.nu)))]
        return ps

    def _read_attributes(self):
        item_attr_reward, attr_reward = self.get_item_attr_reward()
        item_dropout = int((1 - self.args.item_rate) * self.user_item.shape[1])
        attr_dropout = int((1 - self.args.item_rate) * len(attr_reward))
        if item_dropout > 0:
            if self.delete_key_term == "low":
                attr_select = np.argsort(attr_reward)[:attr_dropout]
            elif self.delete_key_term == "random":
                attr_select = np.random.choice(attr_reward.shape[0], attr_dropout, replace=False)
            else:
                item_select = np.argsort(item_attr_reward)[-item_dropout:]
                attr_select = np.argsort(attr_reward)[-attr_dropout:]
        else:
            item_select = []
            attr_select = []

        suparms, user_attr, relation = [], [], dict()
        file_name = self.file_name + '/arm_suparm_relation.txt'
        with open(file_name, 'r') as fr:
            for index, line in enumerate(fr):
                # if index in item_select:
                #     continue
                e = line.strip().split('\t')
                a_id, s_ids = int(e[0]), e[1].strip(', ').split(',')
                w = 1.0 / len(s_ids)
                for key in s_ids:
                    key = int(key)
                    if key in relation:
                        relation[key][a_id] = w
                    else:
                        relation[key] = {a_id: w}

        for index, (s_id, arms) in enumerate(relation.items()):
            if int(s_id) in attr_select:
                continue
            rw = torch.zeros((self.user_item.shape[0],), device=device)
            fv = torch.zeros((self.d,), device=device)
            sum_w = 0
            for a_id, w in arms.items():
                rw += self.user_item[:, a_id] * w
                fv += self.AM.arms[a_id].fv * w
                sum_w += w
            if sum_w > 0:
                user_attr.append(rw / sum_w)
                suparms.append(fv / sum_w)
        # print("user_attr.shape", len(user_attr))
        # input()
        return user_attr, suparms

    def get_item_attr_reward(self):
        suparms, user_attr, relation = [], [], dict()
        file_name = self.file_name + '/arm_suparm_relation.txt'
        with open(file_name, 'r') as fr:
            for index, line in enumerate(fr):
                e = line.strip().split('\t')
                a_id, s_ids = int(e[0]), e[1].strip(', ').split(',')
                w = 1.0 / len(s_ids)
                for key in s_ids:
                    key = int(key)
                    if key in relation:
                        relation[key][a_id] = w
                    else:
                        relation[key] = {a_id: w}

        s_ids = []
        for index, (s_id, arms) in enumerate(relation.items()):
            rw = torch.zeros((self.user_item.shape[0],), device=device)
            fv = torch.zeros((self.d,), device=device)
            sum_w = 0
            for a_id, w in arms.items():
                rw += self.user_item[:, a_id] * w
                fv += self.AM.arms[a_id].fv * w
                sum_w += w
            if sum_w > 0:
                user_attr.append(rw / sum_w)
                suparms.append(fv / sum_w)
            s_ids.append(s_id)
        attr_reward = torch.stack(user_attr).mean(dim=1).cpu().numpy()[np.argsort(s_ids)]
        item_attr_reward = []

        suparms, user_attr, relation = [], [], dict()
        file_name = self.file_name + '/arm_suparm_relation.txt'
        with open(file_name, 'r') as fr:
            for index, line in enumerate(fr):
                e = line.strip().split('\t')
                a_id, s_ids = int(e[0]), e[1].strip(', ').split(',')
                w = 1.0 / len(s_ids)
                item_attr_reward.append(np.mean([attr_reward[int(key)] for key in s_ids]))
                for key in s_ids:
                    key = int(key)
                    if key in relation:
                        relation[key][a_id] = w
                    else:
                        relation[key] = {a_id: w}
        return np.asarray(item_attr_reward), attr_reward
