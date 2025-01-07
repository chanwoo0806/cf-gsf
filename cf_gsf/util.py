import os
import torch
import random
import logging
import importlib
import numpy as np
from scipy.special import comb
from cf_gsf.configurator import args

def init_seed():
    if args.rand_seed is not None:
        seed = args.rand_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
def build_model(data_handler):
    module_path = f'cf_gsf.model.{args.model}'
    module = importlib.import_module(module_path)
    for attr in dir(module):
        if attr.lower() == args.model.lower():
            return getattr(module, attr)(data_handler)
    raise NotImplementedError(f'No model named {args.model} in {module_path}')
    
def build_evaluator(data_handler, logger):
    if args.evaluator is None:
        module_path = 'cf_gsf.evaluator'
        module = importlib.import_module(module_path)
        return getattr(module, 'BaseEvaluator')(data_handler, logger)
    else:
        module_path = f'cf_gsf.evaluator.{args.evaluator}'
        module = importlib.import_module(module_path)
        for attr in dir(module):
            if attr.lower() == args.evaluator.lower():
                return getattr(module, attr)(data_handler, logger)
        raise NotImplementedError(f'No evaluator named {args.evaluator} in {module_path}')

def log_exceptions(func):
    def wrapper(*args, **kwargs):
        logger = logging.getLogger('eval_logger')
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(e)
            raise e
    return wrapper

class Logger:
    def __init__(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO) 
        # Print log to both file and console
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        file_handler = logging.FileHandler(f'{args.path}/eval.log', 'a', encoding='utf-8')
        strm_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(message)s')
        for handler in (file_handler, strm_handler):
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.log_config()

    def log(self, message):
        self.logger.info(message)
      
    def log_config(self):
        config = '\n[CONFIGS]\n'
        for arg, value in args.__dict__.items():
            if value is not None:
                config += f'>>> {arg}: {value}\n'
        self.logger.info(config)

    def log_eval(self, eval_result, ks):
        message = '\n'
        header, values = '', ''
        for metric, result in eval_result.items():
            for i, k in enumerate(ks):
                metric_name = f'{metric}@{k}'
                header += f'{metric_name:>16s}'
                values += f'{result[i]:>16.4f}'
        message += (header + '\n' + values)
        self.logger.info(message)
        
    def log_summary(self, eval_result, ks):
        with open(f'./log/{args.summary}.csv', 'a') as f:
            message = f'{args.comment},'
            for result in eval_result.values():
                for i in range(len(ks)):
                    message += f'{result[i]:.4f},'
            f.write(message[:-1] + '\n')

def scipy_coo_to_torch_sparse(mat):
    # scipy.sparse.coo_matrix -> torch.sparse.FloatTensor
    idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
    vals = torch.from_numpy(mat.data.astype(np.float32))
    shape = torch.Size(mat.shape)
    return torch.sparse.FloatTensor(idxs, vals, shape).to(args.device)

def cheby(k, x):
    if k == 0:
        return 1 if not isinstance(x, np.ndarray) else np.ones_like(x)
    elif k == 1:
        return x
    else:
        return 2 * x * cheby(k-1, x) - cheby(k-2, x)
    
def bern(K, k, x):
    return comb(K, k) * (x**k) * ((1-x)**(K-k))