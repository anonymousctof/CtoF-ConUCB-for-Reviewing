import torch
import numpy as np
import random
from conf import seeds_set

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = seeds_set[-1]
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def edge_probability(n):
    return 3 * np.log(n) / n


def is_power2(n):
    return n > 0 and ((n & (n - 1)) == 0)


def factT(T):
    return np.sqrt((1 + np.log(1 + T)) / (1 + T))


def evolve(T, t, N, delta):
    return min(int(np.log(1 + t / T * delta) / np.log(2) * N) + 1, N)

def evolve_2(T, t, N, delta, lower):
    return min(int(np.log(1 + t * delta) / np.log(lower)) + 1, N)

def evolve_3(T, t, N, delta, lower):
    return min(int(np.log(1 + t / T * (lower - 1) * delta) / np.log(lower) * 100) + 1, N)


class KMeans:
    def __init__(self, n_clusters, random_state, select=None, X_dist=None, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.select = select
        self.max_iter = max_iter
        self.tol = tol
        self.labels_ = None
        self.select_ = None
        self.X_dist = X_dist

    def _initial(self, X):
        if self.select is None:
            select = torch.randint(0, self.n_samples, (1,), device=device)
        else:
            select = torch.tensor(self.select.tolist(), device=device).long()
        if self.X_dist is None:
            self.X_dist = torch.cdist(X, X)
        centers = torch.zeros((self.n_clusters, self.n_features), device=device)
        dist = torch.zeros((self.n_clusters, self.n_samples), device=device)

        for i in range(self.n_clusters):
            centers[i] = X.index_select(0, select)
            dist[i] = self.X_dist.index_select(0, select)
            if i == 0:
                minimum = dist[0]
            else:
                minimum = torch.min(dist[i], minimum)
            select = torch.argmax(minimum)
        return centers

    def fit_predict(self, X, gamma=None):
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.batch = self.n_samples * self.n_clusters // 200000001 + 1

        labels, select, weights = None, None, None

        centers = self._initial(X)
        x = torch.arange(self.n_samples, device=device)
        ones = torch.ones_like(x).float()
        for _iter in range(self.max_iter):
            labels = torch.cat([torch.argmin(torch.cdist(bx, centers), dim=1).view(-1) for bx in X.chunk(self.batch)])
            select = torch.sparse_coo_tensor(torch.stack([labels, x]), ones, (self.n_clusters, self.n_samples))
            weights = torch.sparse.sum(select, dim=1).to_dense()
            if len(weights) == 1:
                new_centers = X.mean(dim=0).view(1, -1)
                print(new_centers)
            new_centers = torch.sparse.mm(select, X) / weights.unsqueeze(1)
            new_centers = new_centers[new_centers[:, 0] > -np.inf]
            self.n_clusters = new_centers.shape[0]

            if centers.shape[0] == new_centers.shape[0]:
                if ((new_centers - centers).norm(dim=1) < self.tol).sum() == self.n_clusters:
                    break
            centers = new_centers.detach().clone()


        if gamma is not None:
            m = torch.nn.Softmax(dim=1)
            dist = torch.cat([m(-gamma * torch.cdist(bx, centers).pow(2)) for bx in X.chunk(self.batch)], dim=0)
            sort = torch.sort(dist, dim=1)
            probs = sort.values.view(-1)
            labels = sort.indices.view(-1)[probs > .95 / self.n_clusters]
            print(len(labels) / self.n_samples)
            x_ind = torch.arange(x.shape[0], device=device).repeat_interleave(
                self.n_clusters)[probs > .95 / self.n_clusters]
            probs = probs[probs > .95 / self.n_clusters]
            ind = torch.stack([labels, x_ind])
            select = torch.sparse_coo_tensor(ind, probs, (self.n_clusters, x.shape[0]))
            try:
                weights = torch.sparse.sum(select, dim=1).to_dense()
            except:
                print(select)

        return labels.cpu().numpy(), select, weights


class GaussianMixture:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_predict(self, x, select, weights, delta=1.e-3, eps=1.e-6, n_iter=100, top=10):  # (n, d,)
        self.n_components = len(weights)
        ind = select.coalesce().indices()
        self.batch = x.shape[0] * self.n_components // 4000001 + 1
        self.log2pi = -.5 * x.shape[1] * np.log(2. * np.pi)
        self.mu = torch.sparse.mm(select, x) / weights.unsqueeze(1)
        self.var = torch.stack([(x.index_select(0, ind[1][ind[0] == i]) - self.mu[i]).pow(2).mean(0)
                                for i in range(self.mu.shape[0])]) + eps
        self.pi = (weights / x.shape[0]).unsqueeze(0) + eps

        probs, self.log_likelihood = self._e_step(x)
        resp = torch.nn.functional.softmax(probs, dim=1)

        for i in range(n_iter):
            log_likelihood_old = self.log_likelihood
            resp_old = resp

            # m_step & update
            self._m_step(x, resp)
            torch.cuda.synchronize()

            # e_step
            probs, self.log_likelihood = self._e_step(x)
            resp = torch.nn.functional.softmax(probs, dim=1)

            # check convergence
            if (self.log_likelihood - log_likelihood_old < delta or
                    ((resp_old - resp).norm(dim=0) < delta).sum() == self.n_components):
                break

        top = min(top, self.n_components)
        probs = torch.nn.functional.softmax(probs, dim=1)
        topk = probs.topk(top, dim=1)
        probs = topk.values.view(-1)
        labels = topk.indices.view(-1)[probs > .1]
        x_ind = torch.arange(x.shape[0], device=device).repeat_interleave(top)[probs > .1]
        probs = probs[probs > .1]
        ind = torch.stack([labels, x_ind])
        select = torch.sparse_coo_tensor(ind, probs, (self.n_components, x.shape[0]))
        try:
            weights = torch.sparse.sum(select, dim=1).to_dense()
        except:
            print(topk)

        return select, weights

    def _m_step(self, x, resp, eps=1.e-6):
        self.pi = torch.sum(resp, dim=0, keepdim=True) + eps
        self.mu = torch.mm(resp.T, x) / self.pi.T
        self.var = torch.stack([
            (r * (bx - self.mu.unsqueeze(0)).pow(2)).sum(0) for r, bx in
            zip(resp.unsqueeze(2).chunk(self.batch), x.unsqueeze(1).chunk(self.batch))
        ]).sum(0) / self.pi.T + eps
        self.pi /= x.shape[0]

    def _e_step(self, x):
        var_log_sum = -.5 * self.var.log().sum(dim=1).unsqueeze(0)
        log_prob = -.5 * torch.cat([((bx.unsqueeze(1) - self.mu.unsqueeze(0)).pow(2) / self.var.unsqueeze(0)).sum(dim=2)
                                    for bx in x.chunk(self.batch)], dim=0) + var_log_sum + self.log2pi + self.pi.log()
        log_likelihood = torch.logsumexp(log_prob, dim=1).mean()  # (n, 1,)
        return log_prob, log_likelihood
