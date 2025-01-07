import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from cf_gsf.configurator import args
from cf_gsf.evaluator import BaseEvaluator
from cf_gsf.util import log_exceptions, cheby, bern

class PolyFilter(BaseEvaluator):
    def __init__(self, data_handler, logger):
        super().__init__(data_handler, logger)
    
    @log_exceptions
    def test(self, model):
        super().test(model)
        self._analyze(model)

    @log_exceptions
    def _analyze(self, model):
        with torch.no_grad():
            # Log filter params    
            self.logger.log(f'[Params] {str_params(model.params)}')
            # Draw filter figs
            if not os.path.exists(f'{args.path}/figs'):
                os.makedirs(f'{args.path}/figs')
            x, bases = get_bases(model.type, args.order)
            self._save_filter_fig(x, bases, model.get_coeffs())
            
    @log_exceptions
    def _save_filter_fig(self, x, bases, weights):
        weights = weights.cpu().numpy()
        f = np.dot(bases, weights)
        b = bases * weights
        weights = [f'[{i}] {w:.2f}' for i, w in enumerate(weights)]
        # Full Function
        plt.plot(x, f)
        plt.axhline(0, color='black', lw=1.5)
        plt.grid(True)
        plt.xlabel('lambda')
        plt.title(' '.join(weights))
        plt.savefig(f'{args.path}/figs/poly.png')
        plt.clf()
        # Each Basis
        plt.plot(x, b)
        plt.axhline(0, color='black', lw=1.5)
        plt.grid(True)
        plt.xlabel('lambda')
        plt.legend(weights)
        plt.savefig(f'{args.path}/figs/bases.png')
        plt.clf()
        # Full with Ideal
        if args.ideal_num:
            path = f'./dataset/{args.dataset}/svd/s.npy'
            if os.path.exists(path):
                s = np.load(f'./dataset/{args.dataset}/svd/s.npy')[args.ideal_num-1].item()
            else:
                norm_inter = self.data_handler.get_normalized_inter()
                s = svds(norm_inter, which='LM', k=args.ideal_num, random_state=args.rand_seed, return_singular_vectors=False)[-1].item()
            freq = 1-s**2 # lambda = 1 - sigma^2 if L = I - R\tilde^T*R\tilde
            i = np.zeros_like(x)
            i[:int(freq*len(i))] = 1
            plt.plot(x, f + args.ideal_weight * i)
            plt.axhline(0, color='black', lw=1.5)
            plt.grid(True)
            plt.xlabel('lambda')
            weights += [f'[Ideal - {args.ideal_num}] {args.ideal_weight:.1f}']
            plt.title(' '.join(weights))
            plt.savefig(f'{args.path}/figs/full.png')
            plt.clf()

def str_params(params):
    params = params.tolist()
    params = [f'({i}) {w:.3f}' for i, w in enumerate(params)]
    params = ' '.join(params)
    return params

def get_bases(poly, order):
    if poly == 'mono':
        f = lambda k, x: x**k
        interval = (0,1)
    elif poly == 'cheb':
        f = lambda k, x: cheby(k, x)
        interval = (-1,1)
    elif poly == 'bern':
        f = lambda k, x: bern(order, k, x)
        interval = (0,1)
    bases = []
    x = np.linspace(*interval, 100)
    for k in range(order+1):
        bases.append(f(k,x))
    y = np.stack(bases, axis=1)
    return x, y