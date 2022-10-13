import math

linucb_para = {'lambda': 1.0, 'sigma': 0.05, 'alpha': 0.25}
conucb_para = {'lambda': 0.5, 'sigma': 0.05, 'tilde_lambda': 1.0, 'alpha': 0.25, 'tilde_alpha': 0.25}
club_para = {'alpha': 0.5}
sclub_para = {'alpha': 0.5, 'alpha_p': math.sqrt(2)}
kmeans_para = {'cluster_num': 10, 'max_iter': 200, 'delta': 1.0}
epsilon = 1e-6
train_iter = 0
test_iter = 1000
armNoiseScale = 0.1
suparmNoiseScale = 0.1
ts_nu = 1.0
batch_size = 50
bt = lambda t: 5 * int(math.log(t + 1))
minRecommend, maxRecommend = 1, 5
seeds_set = [2756048, 675510, 807110, 2165051, 9492253, 927, 218, 495, 515, 452]
