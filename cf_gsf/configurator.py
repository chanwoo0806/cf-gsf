import time
import yaml
import torch
import argparse

def configurate():
    parser = argparse.ArgumentParser()
    
    ### Basic
    parser.add_argument('--default', type=str, default='default')
    parser.add_argument('--rand_seed', type=int)

    ### Logging
    parser.add_argument('--comment', type=str)
    parser.add_argument('--folder', type=str)
    parser.add_argument('--summary', type=str)
    
    ### Test
    parser.add_argument('--metrics', type=str)
    parser.add_argument('--ks', type=str)
    parser.add_argument('--tst_batch', type=int)

    ### Essential
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--evaluator', type=str)
    
    ### GFCF & PGSP
    parser.add_argument('--ideal', type=float, help='weight of ideal low-pass filter')
    
    ### PolyFilter
    parser.add_argument('--order', type=int)
    parser.add_argument('--weights', type=str)
    parser.add_argument('--ideal_num', type=int)
    parser.add_argument('--ideal_weight', type=float)
    
    ### Normalization
    parser.add_argument('--pre', type=str)
    parser.add_argument('--pre_val', type=float)
    parser.add_argument('--post', type=str)
    parser.add_argument('--post_val', type=float)
                                                                                                     
    args = parser.parse_args()
    
    # Use default values if args are not given
    with open(f'./config/{args.default}.yml', mode='r', encoding='utf-8') as f:
        default = yaml.safe_load(f.read())
    for arg, value in args.__dict__.items():
        if (value is None) and (arg in default):
            setattr(args, arg, default[arg])
    
    # Convert comma-separated string to list
    def str_to_list(string, elem_type):
        return [elem_type(x) for x in string.split(",")]
    def is_str(x):
        return isinstance(x, str)
    args.metrics = str_to_list(args.metrics, str) if is_str(args.metrics) else args.metrics
    args.ks = str_to_list(args.ks, int) if is_str(args.ks) else args.ks
    args.weights = str_to_list(args.weights, float) if is_str(args.weights) else args.weights
    
    # Automatically set args
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.path = f'./log/' if args.folder is None else f'./log/{args.folder}/'
    args.path += f'{time.strftime("%m%d-%H%M%S")}-{args.comment}'
    
    return args

args = configurate()

''' <<< Tip >>> Arguments can be set without using command line (in case of Jupyter Notebook).
import sys
sys.argv = ['configurator.py', '--comment', 'jupyter', '--dataset', 'gowalla']
'''