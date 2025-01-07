import torch
from torch import spmm
from cf_gsf.configurator import args
from cf_gsf.model import PolyFilter

class PF_Mono(PolyFilter): # Support: [0,1]
    def __init__(self, data_handler):
        super().__init__(data_handler)
        self.type = 'mono'

    def L_tilde(self, norm_inter, x):
        # L_tilde = L/lambda_max = L/1 = L = I - R_tilde^T*R_tilde
        y = spmm(norm_inter, x)
        y = spmm(norm_inter.t(), y) * (-1)
        y += x
        return y

    def get_bases(self, signal, norm_inter):
        bases = []
        bases.append(signal) # x[0]
        for _ in range(1, args.order+1):
            # x[k] = L * x[k-1]
            basis = self.L_tilde(norm_inter, bases[-1])
            bases.append(basis)
        return torch.stack(bases, dim=0)

    def get_coeffs(self):
        return self.params