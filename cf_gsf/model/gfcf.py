import torch
from torch import spmm
from cf_gsf.model import BaseModel
from cf_gsf.configurator import args

class GFCF(BaseModel):
    def __init__(self, data_handler):
        super().__init__(data_handler)        
        self.data_handler = data_handler
        self.data_handler.pf_set_inter()
        if args.ideal_weight:
            args.ideal_num = 256
            self.data_handler.pf_set_ideal()
        self.pre = self.data_handler.pf_get_prepost('power', -0.5)
        self.post = self.data_handler.pf_get_prepost('power', 0.5)
            
    def _get_signal(self, pck_users):
        pck_users = pck_users.long().cpu().numpy()
        inter = self.data_handler.inter
        signal = torch.tensor(inter[pck_users].todense()).to(args.device).T 
        return signal # (N,B)

    def _get_norm_inter(self):
        norm_inter = self.data_handler.norm_inter
        return norm_inter # (M,N)

    def _get_ideal(self): 
        return self.data_handler.ideal # (N,ideal_num)
    
    def forward(self, pck_users):        
        signal = self._get_signal(pck_users) # (N,B)
        norm_inter = self._get_norm_inter() # (M,N)
        # linear filtering
        full_preds = spmm(norm_inter, signal) # (M,B)
        full_preds = spmm(norm_inter.t(), full_preds) # (N,B)
        full_preds = full_preds.T # (B,N)
        # ideal pass filtering
        if args.ideal_weight:
            ideal = self._get_ideal() # (N,ideal_num)
            ideal_preds = signal.T * self.pre @ ideal @ ideal.T * self.post # (B,N)
            full_preds += args.ideal_weight * ideal_preds
        return full_preds # (B,N)
    
    def full_predict(self, batch_data):
        pck_users, train_mask = batch_data
        full_preds = self.forward(pck_users)
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds