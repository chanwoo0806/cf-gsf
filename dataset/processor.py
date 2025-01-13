import os
import time
import pickle
import argparse
import numpy as np
import scipy.sparse as sp
from scipy.linalg import svd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds

def load_inter(dataset):
    """Load one single adjacent matrix from file
    Args:
        file (string): path of the file to load
    Returns:
        scipy.sparse.coo_matrix: the loaded adjacent matrix
    """
    file = f'./{dataset}/train_mat.pkl'
    with open(file, 'rb') as fs:
        inter = (pickle.load(fs) != 0).astype(np.float32)
    if type(inter) != coo_matrix:
        inter = coo_matrix(inter)
    return inter

def normalize(inter):
    user_degree = np.array(inter.sum(axis=1)).flatten() # Du
    item_degree = np.array(inter.sum(axis=0)).flatten() # Di
    user_d_inv_sqrt = sp.diags(np.power(user_degree + 1e-10, -0.5)) # Du^(-0.5)
    item_d_inv_sqrt = sp.diags(np.power(item_degree + 1e-10, -0.5)) # Di^(-0.5)
    norm_inter = (user_d_inv_sqrt @ inter @ item_d_inv_sqrt)
    return norm_inter

def svd_solver(args):    
    inter = load_inter(args.dataset)
    norm_inter = normalize(inter)
    
    start_time = time.time()
    if args.full:
        print(">>> Full SVD")
        u, s, vt = svd(norm_inter.todense(), full_matrices=False, lapack_driver='gesvd')
        v = vt.T
    else:
        print(f">>> Truncated SVD - Cutoff: {args.cutoff}, Random seed: {args.rand_seed}")
        u, s, vt = svds(norm_inter, which='LM', k=args.cutoff, random_state=args.rand_seed)
        v = vt.T
        u = np.flip(u, axis=1)
        v = np.flip(v, axis=1)
        s = np.flip(s)
    print(f">>> Computation for {(time.time() - start_time)/60:.1f} mins")
    
    if not os.path.exists(f'./{args.dataset}/svd'):
        os.makedirs(f'./{args.dataset}/svd')
    
    name = 'full' if args.full else str(args.cutoff)    
    np.save(f'./{args.dataset}/svd/u_{name}.npy', u)
    np.save(f'./{args.dataset}/svd/s_{name}.npy', s)
    np.save(f'./{args.dataset}/svd/v_{name}.npy', v)
    
def read_cf(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    max_uid, max_iid = 0, 0
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]
        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        max_uid = max(max_uid, u_id)
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])
            max_iid = max(max_iid, i_id)
    return np.array(inter_mat), max_uid, max_iid

def get_sp_mat(cf_data, max_uid, max_iid):
    ui_edges = list()
    for u_id, i_id in cf_data:
        ui_edges.append([u_id, i_id])
    ui_edges = np.array(ui_edges)
    vals = [1.] * len(ui_edges)
    mat = sp.coo_matrix((vals, (ui_edges[:, 0], ui_edges[:, 1])), shape=(max_uid+1, max_iid+1))
    return mat

def txt2pkl(args):
    split = ('train', 'test')
    total = 0
    for s in split:
        cf, max_uid, max_iid = read_cf(f"./{args.dataset}/raw/{s}.txt")
        mat = get_sp_mat(cf, max_uid, max_iid)
        pickle.dump(mat, open(f"./{args.dataset}/{s}_mat.pkl", 'wb'))
        print(f"# of interaction in {s}:", mat.count_nonzero())
        total += mat.count_nonzero()
    print(f"# of total interaction:", total)
    print(f"# of users: {max_uid+1}, # of items: {max_iid+1}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--rand_seed', type=int, default=2024)
    # SVD
    parser.add_argument('--full', action='store_true') # full svd on dense matrix
    parser.add_argument('--cutoff', type=int) # truncated svd on sparse matrix
    
    args = parser.parse_args()
    print(f">>> Dataset: {args.dataset}")
    if args.task == 'svd':
        svd_solver(args)
    elif args.task == 'txt2pkl':
        txt2pkl(args)