import os
import torch
import numpy as np
from torch import spmm
from scipy.sparse.linalg import svds
from cf_gsf.model import BaseModel
from cf_gsf.configurator import args

class PGSP(BaseModel):
    def __init__(self, data_handler):
        super().__init__(data_handler)
        self.data_handler = data_handler
        self.data_handler.pf_set_inter()
        self.ideal_num = 512
        self.set_filter()
        
    def set_filter(self):
        # set R_tilde @ R_tilde^T for signal_u
        norm_inter = self.data_handler.get_normalized_inter()
        self.norm_item_co = (norm_inter @ norm_inter.transpose()).tocsr()
        # set ideal pass filter
        if args.ideal_weight:
            path = f'./dataset/{args.dataset}/svd'
            if os.path.exists(path):
                u = np.load(f'{path}/u.npy')[:, :self.ideal_num]
                v = np.load(f'{path}/v.npy')[:, :self.ideal_num]
            else:
                u, s, v = svds(self.norm_inter, which='LM', k=self.ideal_num, random_state=args.rand_seed)
                v = v.T
            self.u, self.v = torch.tensor(u).to(args.device), torch.tensor(v).to(args.device)
        # set pre/post processor
        if args.pre_val:
            norm_item_co_degree = np.array(self.norm_item_co.sum(axis=1)).flatten()
            self.pre_u = torch.tensor(norm_item_co_degree + 1e-10).to(args.device).pow(args.pre_val)
            self.pre_i = self.data_handler.pf_get_prepost('power', args.pre_val)
            self.post_i = self.data_handler.pf_get_prepost('power', -args.pre_val)

    def _get_signal(self, pck_users):
        pck_users = pck_users.long().cpu().numpy()
        # signal_u
        norm_item_co = self.norm_item_co
        signal_u = torch.tensor(norm_item_co[pck_users].todense()).to(args.device).T # (M, B)
        # signal_i
        inter = self.data_handler.inter
        signal_i = torch.tensor(inter[pck_users].todense()).to(args.device).T  # (N, B)
        return signal_u, signal_i

    def _get_norm_inter(self):
        norm_inter = self.data_handler.norm_inter
        return norm_inter # (M,N)

    def forward(self, pck_users):
        signal_u, signal_i = self._get_signal(pck_users) # (M, B), (N, B)
        norm_inter = self._get_norm_inter() # (M,N)
        # preprocessing
        if args.pre_val:
            signal_u = signal_u * self.pre_u.reshape(-1,1) # (M, B)
            signal_i = signal_i * self.pre_i.reshape(-1,1) # (N, B)
        # linear filtering - u
        preds_u = spmm(norm_inter.t(), signal_u) # (N, B)
        preds_u = preds_u.T # (B, N)
        # linear filtering - i
        preds_i  = spmm(norm_inter, signal_i) # (M, B)
        preds_i  = spmm(norm_inter.t(), preds_i) # (N, B)
        preds_i  = preds_i.T # (B, N)   
        if args.ideal_weight:
            # ideal pass filtering - u
            ideal_preds_u = signal_u.T @ self.u @ self.v.T # (B, N)
            preds_u =  (1-args.ideal_weight) * preds_u + args.ideal_weight * ideal_preds_u
            # ideal pass filtering - i
            ideal_preds_i = signal_i.T @ self.v @ self.v.T # (B, N)
            preds_i =  (1-args.ideal_weight) * preds_i + args.ideal_weight * ideal_preds_i
        # combine u and i
        full_preds = preds_u + preds_i
        # postprocessing
        if args.pre_val:
            full_preds = full_preds * self.post_i.reshape(1,-1)
        return full_preds # (B,N)

    def full_predict(self, batch_data):
        pck_users, train_mask = batch_data
        full_preds = self.forward(pck_users)
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds        