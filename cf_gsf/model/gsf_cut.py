import numpy
import torch
from cf_gsf.model import BaseModel
from cf_gsf.configurator import args

class GSF_Cut(BaseModel):
    def __init__(self, data_handler):
        super().__init__(data_handler)        
        self.inter = data_handler.get_inter().tocsr()
        self.cutoff = args.cutoff
        self.cutoff_how = args.cutoff_how
        self.pre = data_handler.pf_get_prepost('power', -args.post_val)
        self.post = data_handler.pf_get_prepost('power', args.post_val)
        self._set_filter()
    
    def _set_filter(self):
        s = numpy.load(f'./dataset/{args.dataset}/svd/s.npy')
        V = numpy.load(f'./dataset/{args.dataset}/svd/v.npy')
        freq = torch.tensor(1 - s**2).to(args.device) # (M,)
        
        # mask
        if self.cutoff_how == 'num':
            mask = torch.zeros_like(freq)
            cutoff_num = int(len(freq) * self.cutoff)
            mask[:cutoff_num] = 1.0
        elif self.cutoff_how == 'val':
            mask = (freq <= self.cutoff).float()
            
        # truncated & V    
        self.truncated = mask # (M,)
        self.V = torch.tensor(V).to(args.device) # (N,M)
            
    def _get_signal(self, pck_users):
        pck_users = pck_users.long().cpu().numpy()
        signal = torch.tensor(self.inter[pck_users].todense()).to(args.device).T 
        return signal # (N,B)
    
    def forward(self, pck_users):        
        signal = self._get_signal(pck_users) # (N,B)
        signal = signal * self.pre.reshape(-1,1) # (N,B)
        signal = self.V.T @ signal # (M,B)
        signal = signal * self.truncated.reshape(-1,1) # (M,B)
        signal = self.V @ signal # (N,B)
        signal = signal * self.post.reshape(-1,1) # (N,B)
        full_preds = signal.T # (B,N)
        return full_preds # (B,N)
    
    def full_predict(self, batch_data):
        pck_users, train_mask = batch_data
        full_preds = self.forward(pck_users)
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds