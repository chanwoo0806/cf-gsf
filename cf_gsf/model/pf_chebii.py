import math
import torch
import torch.nn.functional as F
from cf_gsf.util import cheby
from cf_gsf.configurator import args
from cf_gsf.model.pf_cheb import PF_Cheb

class PF_ChebII(PF_Cheb): # Support: [-1,1]
    def __init__(self, data_handler):
        super().__init__(data_handler)

    def get_coeffs(self):
        coeffs = []
        K = args.order
        for k in range(K+1):
            coeff = torch.tensor(0.0).to(args.device)
            for j in range(K+1):
                x_j = math.cos((K-j+0.5) * math.pi / (K+1)) # Chebyshev node
                coeff += self.params[j] * cheby(k,x_j)
            coeff *= (2/(K+1)) 
            coeffs.append(coeff)
        coeffs[0] /= 2 # the first term is to be halved
        return torch.stack(coeffs)