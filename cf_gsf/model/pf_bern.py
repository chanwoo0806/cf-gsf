import torch
from torch import spmm
from scipy.special import comb
from cf_gsf.configurator import args
from cf_gsf.model import PolyFilter

class PF_Bern(PolyFilter): # Support: [0,1]
    def __init__(self, data_handler):
        super().__init__(data_handler)
        self.type = 'bern'

    def L_tilde(self, norm_inter, x):
        # L_tilde = L/lambda_max = L/1 = L = I - R_tilde^T*R_tilde
        y = spmm(norm_inter, x)
        y = spmm(norm_inter.t(), y) * (-1)
        y += x
        return y
    
    def I_minus_L_tilde(self, norm_inter, x):
        # I - L_tilde = I - L = R_tilde^T*R_tilde
        x = spmm(norm_inter, x)
        x = spmm(norm_inter.t(), x)
        return x

    def get_bases(self, signal, norm_inter):
        # (I-L_tilde)^k
        bases = []
        bases.append(signal)
        for _ in range(1, args.order+1):
            basis = self.I_minus_L_tilde(norm_inter, bases[-1])
            bases.append(basis)
        bases.reverse()
        # (L_tilde)^k
        for k in range(args.order+1):
            basis = bases[k]
            for _ in range(k):
                basis = self.L_tilde(norm_inter, basis)
            bases[k] = basis
        # Binomial coeffs
        for k in range(args.order+1):
            bases[k] *= comb(args.order, k)
        return torch.stack(bases, dim=0)

    def get_coeffs(self):
        return self.params
    