import os
import time
import random
import pickle
import argparse
import warnings
import numpy as np
import scipy.sparse as sp
from collections import Counter
from sklearn import cluster
from scipy.sparse.linalg import norm
from multiprocessing import Pool
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')


def set_seed(seed):
    print('Unfolder Set Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)


def compute_adj_element(l):
    adj_map = NUM_NODES + np.zeros((l[1] - l[0], args.max_nei), dtype=np.int)
    sub_adj = RAW_ADJ[l[0]: l[1]]
    for v in range(l[0], l[1]):
        neighbors = np.nonzero(sub_adj[v - l[0], :])[1]
        len_neighbors = len(neighbors)
        if len_neighbors > args.max_nei:
            if args.sorted:
                weight_sort = np.argsort(-sub_adj[v - l[0], neighbors].toarray()).reshape(-1)
                neighbors = neighbors[weight_sort[: args.max_nei]]
            else:
                neighbors = np.random.choice(neighbors, args.max_nei, replace=False)
            adj_map[v - l[0]] = neighbors
        else:
            adj_map[v - l[0], :len_neighbors] = neighbors
    return adj_map


def compute_adjlist_parallel(sp_adj, batch=50):
    global RAW_ADJ
    RAW_ADJ = sp_adj
    index_list = []
    for ind in range(0, NUM_NODES, batch):
        index_list.append([ind, min(ind + batch, NUM_NODES)])
    with Pool(args.n_jobs) as pool:
        adj_list = pool.map(compute_adj_element, index_list)
    adj_list.append(NUM_NODES + np.zeros((1, args.max_nei), dtype=np.int))
    adj_map = np.vstack(adj_list)
    return adj_map


def get_traj_child(parent, sample_num=0):
    '''
    If sample_num == 0 return all the neighbors
    '''

    traj_list = []
    for p in parent:
        neigh = np.unique(ADJ_TAB[p].reshape([-1]))
        if len(neigh) > 1:
            neigh = neigh[neigh != NUM_NODES]
        neigh = np.random.choice(neigh, min(args.max_nei, len(neigh)), replace=False)
        t_array = np.hstack(
            [p * np.ones((len(neigh), 1)).astype(np.int), neigh.reshape([-1, 1])])
        traj_list.append(t_array)
    traj_array = np.unique(np.vstack(traj_list), axis=0)
    if traj_array.shape[0] > 1:
        traj_array = traj_array[traj_array[:, -1] != NUM_NODES]
    if sample_num:
        traj_array = traj_array[
            np.random.choice(
                traj_array.shape[0], min(sample_num, traj_array.shape[0]), replace=False)]
    return traj_array


def get_traj(idx):
    '''
    Get the trajectory set of a given node under the naive setting.
    '''
    traj_list = [np.array(idx), []]
    whole_trajs = np.unique(ADJ_TAB[idx])
    for _ in range(args.K - 1):
        whole_trajs = get_traj_child(whole_trajs, 0)
    traj_list[1] = [whole_trajs]
    return traj_list


def get_cf_score(idx):
    traj = traj_list[idx]
    cen_node = traj[0]
    traj_idx = traj[1][0]
    P = NVS[cen_node]
    Q = NVS[traj_idx[:, 0]]
    R = NVS[traj_idx[:, 1]]
    p_q = np.mean(np.abs(P - Q), axis=1)
    q_r = np.mean(np.abs(Q - R), axis=1)
    traj_sim = p_q + q_r
    return np.array([np.mean(p_q), np.mean(q_r), np.mean(traj_sim)])


def infomax_samp_traj(traj):
    n_samp1 = args.sep_samp[0]
    n_samp2 = args.sep_samp[1]
    cen_node = traj[0]
    traj_vec = np.array([cen_node])
    traj_idx = traj[1][0]

    ### Blank Graph
    if np.sum(traj_idx[:, -1] != NUM_NODES) == 0:
        traj_vec = np.hstack([
            traj_vec,
            np.ones(n_samp1).astype(int) * cen_node,
            np.ones(n_samp2).astype(int) * cen_node])
        return (traj_vec)

    ### Sample of First-ord neis
    traj_idx = traj_idx[traj_idx[:, -1] != NUM_NODES]
    idxes = np.unique(traj_idx[:, 0].reshape([-1]))
    if len(idxes) > 1:
        idxes = idxes[idxes != cen_node]
    if len(idxes) >= n_samp1:
        P = NVS[cen_node]
        Q = NVS[idxes]
        traj_sim = np.sum(P * Q, axis=1)
        traj_rank = np.argsort(traj_sim)
        idxes1 = idxes[traj_rank[-n_samp1:]]
    else:
        extra_idxes = np.random.choice(idxes, n_samp1 - len(idxes))
        idxes1 = np.hstack([idxes, extra_idxes])
    traj_vec = np.hstack([traj_vec, idxes1])

    ### Sample of second-ord neis
    idxes = np.unique(traj_idx[:, -1].reshape([-1]))
    if len(idxes) > 1:
        idxes = idxes[idxes != cen_node]
    if len(idxes) <= n_samp2:
        extra_idxes = np.random.choice(idxes, n_samp2 - len(idxes))
        idxes2 = np.hstack([idxes, extra_idxes])
    else:
        P = NVS[cen_node]
        Q = NVS[idxes]
        traj_sim = np.sum(P * Q, axis=1)
        traj_rank = np.argsort(traj_sim)
        idxes2 = idxes[traj_rank[-n_samp2:]]
    traj_vec = np.hstack([traj_vec, idxes2])
    return traj_vec


INIT_FLAG = True
DATASET = ''
EMB_METH = ''


def samp(args, node_type, samp_meth, samp_num):
    global DATASET, EMB_METH
    if (args.dataset != DATASET) or (args.emb != EMB_METH) or INIT_FLAG:

        globals()['args'] = args
        args.K = 2
        set_seed(args.seed)
        print(args.dataset)
        DATASET = args.dataset
        EMB_METH = args.emb
        globals()['RAW_ADJ'] = sp.load_npz(args.datadir + args.dataset + '_adj_train.npz')
        globals()['NUM_NODES'] = RAW_ADJ.shape[0]
        globals()['FEATURES'] = np.load(args.datadir + args.dataset + '_{}.npy'.format(args.emb))
        null_feature = np.zeros((1, FEATURES.shape[1]))
        globals()['FEATURES'] = np.vstack([FEATURES, null_feature])
        globals()['ADJ_TAB'] = compute_adjlist_parallel(RAW_ADJ)
        globals()['NORM_ADJ'] = normalize(RAW_ADJ, axis=1, norm='l1')
        globals()['NVS'] = FEATURES
        map_dict = pickle.load(open(args.datadir + args.dataset + '_map.pkl', 'rb'))
        user_num = map_dict['user_num']
        item_num = map_dict['item_num']

        globals()['user_list'] = list(range(user_num))
        globals()['item_list'] = list(range(user_num, user_num + item_num))
        node_list = user_list + item_list

        if args.sorted:
            sort_str = 'sort'
        else:
            sort_str = 'unsort'
        traj_file = args.trajdir + args.dataset + '_' + sort_str + '_traj.pkl'

        if args.recompute or not os.path.exists(traj_file):
            t0 = time.time()
            with Pool(args.n_jobs) as pool:
                user_traj = pool.map(get_traj, user_list)
                item_traj = pool.map(get_traj, item_list)
            t1 = time.time()
            globals()['TRAJS'] = user_traj + item_traj
            pickle.dump(TRAJS, open(traj_file, 'wb'))
            print('Dumped Trajs in %.2f second, saving to %s' % (time.time() - t1, traj_file))
        else:
            globals()['TRAJS'] = pickle.load(open(traj_file, 'rb'))
            print('Load stored trajs %s' % (traj_file))
        print('Init Global Variables!')
        globals()['INIT_FLAG'] = False
    else:
        print('Skip Initialization!')

    args.sep_samp = samp_num
    t0 = time.time()
    with Pool(args.n_jobs) as pool:
        samps = np.stack(pool.map(eval(samp_meth + '_samp_traj'), [TRAJS[_] for _ in eval(node_type + '_list')]))
    t1 = time.time()
    print('Sampling %s Trajs in %.2f seconds' % (node_type, t1 - t0))
    return samps
