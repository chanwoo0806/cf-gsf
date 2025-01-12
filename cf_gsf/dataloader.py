import os
import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
import torch
import torch.utils.data as data
from cf_gsf.configurator import args
from cf_gsf.util import scipy_coo_to_torch_sparse

class AllRankTstData(data.Dataset):
    def __init__(self, tst_mat, trn_mat):
        self.trn_mat = (trn_mat.tocsr() != 0) * 1.0
        user_pos_lists = [list() for _ in range(tst_mat.shape[0])]
        test_users = set()
        for i in range(len(tst_mat.data)):
            row = tst_mat.row[i]
            col = tst_mat.col[i]
            user_pos_lists[row].append(col)
            test_users.add(row)
        self.test_users = np.array(list(test_users))
        self.user_pos_lists = user_pos_lists
    
    def __len__(self):
        return len(self.test_users)
    
    def __getitem__(self, idx):
        pck_user = self.test_users[idx]
        pck_mask = self.trn_mat[pck_user].toarray()
        pck_mask = np.reshape(pck_mask, [-1])
        return pck_user, pck_mask

class DataHandler:
    def __init__(self):
        predir = f'./dataset/{args.dataset}/'
        self.trn_file = predir + 'train_mat.pkl'
        self.tst_file = predir + 'test_mat.pkl'

    def _load_one_mat(self, file):
        """Load one single adjacent matrix from file
        Args:
            file (string): path of the file to load
        Returns:
            scipy.sparse.coo_matrix: the loaded adjacent matrix
        """
        with open(file, 'rb') as fs:
            mat = (pickle.load(fs) != 0).astype(np.float32)
        if type(mat) != coo_matrix:
            mat = coo_matrix(mat)
        return mat
    
    def load_data(self):
        trn_mat = self._load_one_mat(self.trn_file)
        self.user_num, self.item_num = trn_mat.shape
        tst_mat = self._load_one_mat(self.tst_file)
        tst_data = AllRankTstData(tst_mat, trn_mat)
        self.test_dataloader  = data.DataLoader(tst_data, batch_size=args.tst_batch, shuffle=False, num_workers=0)

    def get_inter(self):
        return self._load_one_mat(self.trn_file)

    def get_normalized_inter(self, inter=None):
        # Get interaction matrix
        inter = self.get_inter() if inter is None else inter # R
        # Normalize interaction matrix
        user_degree = np.array(inter.sum(axis=1)).flatten() # Du
        item_degree = np.array(inter.sum(axis=0)).flatten() # Di
        user_d_inv_sqrt = sp.diags(np.power(user_degree + 1e-10, -0.5)) # Du^(-0.5)
        item_d_inv_sqrt = sp.diags(np.power(item_degree + 1e-10, -0.5)) # Di^(-0.5)
        norm_inter = (user_d_inv_sqrt @ inter @ item_d_inv_sqrt).tocoo() # Du^(-0.5) * R * Di^(-0.5)
        return norm_inter

    def pf_set_inter(self):
        self.inter = self.get_inter().tocsr() # scipy.sparse.csr
        norm_inter = self.get_normalized_inter()
        self.norm_inter = scipy_coo_to_torch_sparse(norm_inter) # torch.sparse.FloatTensor
    
    def pf_set_ideal(self):
        path = f'./dataset/{args.dataset}/svd/v.npy'
        if os.path.exists(path):
            v = np.load(path)[:, :args.ideal_num] # caveat: singular vectors need to be sorted in descending order
            self.ideal = torch.tensor(v).to(args.device)
        else:
            norm_inter = self.get_normalized_inter()
            u, s, v = svds(norm_inter, which='LM', k=args.ideal_num, random_state=args.rand_seed) # SVD for k largest singular vals
            self.ideal = torch.tensor(v.T.copy()).to(args.device)
            
    def pf_get_prepost(self, method, val):
        inter = self.get_inter()
        item_degree = torch.tensor(np.array(inter.sum(axis=0)).flatten()).to(args.device)
        if method == 'power':
            return (item_degree + 1e-10).pow(val)