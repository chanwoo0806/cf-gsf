import torch
from torch import spmm
from cf_gsf.configurator import args
from cf_gsf.model import PolyFilter

class PF_Cheb(PolyFilter): # Support: [-1,1]
    def __init__(self, data_handler):
        super().__init__(data_handler)
        self.type = 'cheb'

    def L_tilde(self, norm_inter, x):
        # L_tilde = 2L/lambda_max - I = 2(I-R_tilde^T*R_tilde)/1 - I = I -2*R_tilde^T*R_tilde
        y = spmm(norm_inter, x)
        y = spmm(norm_inter.t(), y) * (-2)
        y += x
        return y

    def get_bases(self, signal, norm_inter):
        bases = []
        bases.append(signal) # x[0]
        bases.append(self.L_tilde(norm_inter, signal)) # x[1]
        for _ in range(2, args.order+1):
            # x[k] = 2 * L_tilde * x[k-1] - x[k-2]
            basis = self.L_tilde(norm_inter, bases[-1]) * 2
            basis -= bases[-2]
            bases.append(basis)
        return torch.stack(bases, dim=0)

    def get_coeffs(self):
        return self.params