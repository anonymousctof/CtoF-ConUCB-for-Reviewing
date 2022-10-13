import numpy as np
import torch
from Agent_topk.LinUCB import LinUCB
import conf
import random
from conf import seeds_set

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ArmCon(LinUCB):
    def __init__(self, nu, d, T, args):
        super(ArmCon, self).__init__(nu, d, T, args)
        # fix random seed
        self.seed = seeds_set[args.seed]
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.alpha = args.alpha if args.alpha is not None else conf.linucb_para['alpha']


    def store_info_suparm(self, i, x, y):
        self._update_inverse(i, x, y, tilde=False)
