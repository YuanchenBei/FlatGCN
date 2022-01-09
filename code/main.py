import os
import sys
import json
import ndcg
import time
import uuid
from model import FlatGCN
import random
import gsamp
import pickle
import argparse
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import pickle as pkl
from os import path
import torch.nn.functional as F
from sklearn import metrics
import multiprocessing as mul


def set_seed(seed, cuda):
    print('Set Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default="sampling",
                    help='Model name')
parser.add_argument('--no_sort', action='store_true', default=True,
                help='Whether use Sampling.')
parser.add_argument('--model', type=str, default="ECC",
                    help='Model name')
parser.add_argument('--dataset', type=str, default="gowalla",
                    help='Dataset to use.')
parser.add_argument('--datadir', type=str, default="../data/",
                    help='Director of the dataset.')
parser.add_argument('--trajdir', type=str, default="../traj/",
                    help='Director of the dataset.')
parser.add_argument('--u_samp', type=str, default='infomax', 
                    help='Sampling method for user.')
parser.add_argument('--i_samp', type=str, default='infomax', 
                    help='Sampling method for item.')
parser.add_argument('--u_size', nargs='?', default='[25,25]',
                    help='User sampling size')
parser.add_argument('--i_size', nargs='?', default='[25,25]',
                    help='Item sampling size')
parser.add_argument('--loss', type=str, default='norm', 
                    help='Loss function.')
parser.add_argument('--emb', type=str, default='lgn', 
                    help='Emebdding method')
parser.add_argument('--samp_size', type=int, default=25,
                    help='Sampling size.')
parser.add_argument('--gun_layer', type=int, default=3,
                    help='network layer num.')
parser.add_argument('--layers', nargs='?', default='[0,1,2]',
                    help='network layers')
parser.add_argument('--n_jobs', type=int, default=8,
                    help='Multiprocessing number.')
parser.add_argument('--neg_num', type=int, default=3,
                    help='BPR negative sampling number.')
parser.add_argument('--smlp_size', type=int, default=256,
                    help='MLP_size of stack machine.')
parser.add_argument('--smlp_ly', type=int, default=3,
                    help='MLP layer num of stack machine.')
parser.add_argument('--if_output', action='store_true', default=True,
                    help='Whether output.')
parser.add_argument('--if_stack', action='store_true', default=False,
                    help='Wether useing stack embeddings not aggregated embeddings.')
parser.add_argument('--batch_size', type=int, default=10240,
                    help='Normal batch size.')
parser.add_argument('--warm_batch_size', type=int, default=256,
                    help='Warming up batch size.')
parser.add_argument('--warm_batch_num', type=int, default=100,
                    help='Batchs of the warming up.')
parser.add_argument('--out_epoch', type=int, default=10,
                    help='Validation per training batch.')
parser.add_argument('--patience', type=int, default=10,
                    help='Early stop patience.')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Whether use CUDA.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Multiprocessing number.')
parser.add_argument('--drop_rate', type=float, default=0.6,
                    help='Drop Rate.')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='Multiprocessing number.')
parser.add_argument('--Ks', nargs='?', default='[20]',
                    help='Output sizes of every layer')
parser.add_argument('--skip', type=int, default=0,
                    help='SKip epochs.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random Seed.')
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU ID.')
parser.add_argument('--max_nei', type=int, default=256, 
                    help='Multiprocessing number.')
parser.add_argument('--recompute', action='store_true', default=False,
                help='Whether recompute the trajectory list.')
args, _ = parser.parse_known_args()
args.layers = eval(args.layers)
args.u_size = eval(args.u_size)
args.i_size = eval(args.i_size)
args.sorted = not args.no_sort

print('#' * 70)
if not args.if_stack:
    args.if_raw = True
if args.if_output:
    print('\n'.join([(str(_) + ':' + str(vars(args)[_])) for _ in vars(args).keys()]))
args.cuda = not args.no_cuda and torch.cuda.is_available()
set_seed(args.seed, args.cuda)
args.device = torch.device("cuda:%d" % args.gpu if args.cuda else "cpu")
print(args.device)
args.loss = 'bpr'
ndcg.init(args)
para_dict = pickle.load(open(args.datadir + args.dataset + '_map.pkl', 'rb'))
uuid_code = str(uuid.uuid4())[:4]
root_path = os.getcwd() + '/'
save_path = root_path + 'model_save/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
save_file = save_path + args.dataset + uuid_code
TR_ITEMS = para_dict['tr_items']
VA_ITEMS = para_dict['va_items']
TS_ITEMS = para_dict['ts_items']
POS_ITEMS = para_dict['pos_items']
USER_NUM = para_dict['user_num']
ITEM_NUM = para_dict['item_num']
VA_LBS = para_dict['va_lbs']
TS_LBS = para_dict['ts_lbs']
SUB_TEST = np.random.choice(list(TS_ITEMS.keys()), 100, replace=False)
USER_ARRAY = np.array(list(range(USER_NUM)))
ITEM_ARRAY = np.array(list(range(ITEM_NUM)))
SAMP_POOL = None


def agg_emb(feat, trajs, samp_num, batch=10000):
    num = trajs.shape[0]
    samp_pos = np.cumsum([0, 1] + samp_num)
    embs = []
    for beg in range(0, num, batch):
        end = min(beg + batch, num)
        sub_trajs = trajs[beg:end]
        sub_embs = []
        for _ in range(len(samp_pos) - 1):
            sub_embs.append(feats[sub_trajs[:, samp_pos[_]:samp_pos[_ + 1]]].mean(1, keepdims=True))
        sub_embs = np.concatenate(sub_embs, axis=1)
        embs.append(sub_embs)

    return np.vstack(embs)


feats = np.load(args.datadir + args.dataset + '_{}.npy'.format(args.emb))
user_samp = gsamp.samp(args, 'user', args.u_samp, args.u_size)
user_emb = agg_emb(feats, user_samp, args.u_size)
item_samp = gsamp.samp(args, 'item', args.i_samp, args.i_size)
item_emb = agg_emb(feats, item_samp, args.i_size)
emb = np.vstack([user_emb, item_emb])


def _vasamp_bpr_pair(udata):
    uid = udata[0]
    np.random.seed(udata[1])
    ret_array = np.zeros((args.neg_num, 3)).astype(np.int)
    pos_train = VA_ITEMS.get(uid, [])
    if len(pos_train) == 0:
        return np.zeros((0, 3)).astype(np.int)
    pos_set = set(POS_ITEMS.get(uid, []))
    samp_pos = np.random.choice(pos_train, 1).astype(np.int)
    neg_items = np.random.choice(ITEM_ARRAY, 5 * args.neg_num)
    samp_neg = np.array(neg_items[[_ not in pos_set for _ in neg_items]])[:args.neg_num].astype(np.int)
    ret_array[:, 0] = uid
    ret_array[:, 1] = samp_pos + USER_NUM
    ret_array[:, 2] = samp_neg + USER_NUM
    return ret_array


def vabpr_generate(num=None):
    global SAMP_POOL
    if not SAMP_POOL:
        SAMP_POOL = mul.Pool(args.n_jobs)
    if not num:
        num = args.batch_size
    samp_user = np.hstack(
        [np.random.choice(USER_ARRAY, num).reshape([-1, 1]), np.random.randint(0, 2 ** 32, num).reshape([-1, 1])])
    bpr_lbs = np.vstack(SAMP_POOL.map(_vasamp_bpr_pair, samp_user))
    return bpr_lbs


def bpr_loss(tr_out):
    tr_out = tr_out.reshape([-1, 2])
    return torch.mean(-torch.log(torch.sigmoid_(tr_out[:, 0] - tr_out[:, 1]) + 1e-9))


def _predict(net, lbs, batch_size=20480):
    net.eval()
    with torch.no_grad():
        out_list = []
        for begin in range(0, lbs.shape[0], batch_size):
            end = min(begin + batch_size, lbs.shape[0])
            batch_lbs = lbs[begin:end, :].copy()
            batch_lbs[:, 1] += USER_NUM
            out = net(batch_lbs)
            out_list.append(out)
        out = torch.cat(out_list, dim=0).cpu().data.numpy().reshape(-1)
    return out

t0 = time.time()
X = torch.from_numpy(emb).float().to(args.device)

layer_pairs = [
    [0, 0],
    [0, 1],
    [0, 2],
    [1, 0],
    [1, 1],
    [1, 2],
    [2, 0],
    [2, 1],
    [2, 2]
]

net = FlatGCN(X, layer_pairs, mlp_size=args.smlp_size, mlp_layer=args.smlp_ly, if_xavier=True,
                                  drop_rate=args.drop_rate, device=args.device)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.weight_decay)
if args.if_output:
    print(net)

loss_func = bpr_loss
patience_count = 0
va_acc_max = 0
ts_score_max = 0
batch = 0
time_list = []
time_plot_list = []
early_stop_flag = False
train_time = 0
opt_time = 0
pure_train_time = 0
batch_size = args.warm_batch_size

for epoch in range(10000):
    if early_stop_flag:
        break
    t_epoch_begin = time.time()
    epoch_train_cost = []
    epoch_val_cost = []
    for ind in range(0, max(int(len(USER_ARRAY) / batch_size), 1)):
        t_train_begin = time.time()
        if batch == args.warm_batch_num:
            batch_size = args.batch_size
            print('*' * 25, 'Updating BatchSize to %d' % (batch_size), '*' * 25)
            batch += 1
            break
        batch += 1
        batch_lbs = vabpr_generate(batch_size)
        batch_lbs = np.hstack([batch_lbs[:, [0, 1]], batch_lbs[:, [0, 2]]]).reshape([-1, 2])
        net.train()
        t_opt_begin = time.time()
        optimizer.zero_grad()
        tr_out = net(batch_lbs)
        loss = loss_func(tr_out)
        loss.backward()
        optimizer.step()
        t_train_end = time.time()
        epoch_train_cost.append(t_train_end - t_train_begin)
        train_time += t_train_end - t_train_begin
        opt_time += t_train_end - t_opt_begin
    if (epoch % args.out_epoch == 0) or (epoch < args.out_epoch):
        net.eval()
        t_val_begin = time.time()
        va_acc = ndcg._auc(net, _predict)
        time_plot_list.append([epoch, train_time, va_acc])
        if epoch > args.skip:
            if va_acc > va_acc_max:
                va_acc_max = va_acc
                torch.save(net.state_dict(), save_file)
                patience_count = 0
            else:
                patience_count += 1
                if patience_count > args.patience:
                    early_stop_flag = True
                    break
        t_val_end = time.time()
        epoch_val_cost.append(t_val_end - t_val_begin)
        if args.if_output:
            print(
                'Epo%d(%d/%d) loss:%.4f|TS_auc:%.4f|BestTS_auc:%.4f|Train:%.2fs,Opt:%.2f' % (
                    epoch + 1, patience_count, args.patience, loss.data, va_acc,
                    va_acc_max, train_time, opt_time))
    t_epoch_end = time.time()
    time_list.append([
        t_epoch_end - t_epoch_begin,
        np.sum(epoch_train_cost),
        np.sum(epoch_val_cost),
        train_time
    ])
print('Training Finished!!')

time_array = np.array(time_list)
t1 = time.time()
net.load_state_dict(torch.load(save_file))
net.eval()
running_cost = t1 - t0
result = [args.dataset + ',' + args.emb, args.u_samp + args.i_samp, epoch]
result += [
    running_cost,
    train_time,
    np.mean(time_array[:, 0]),
    np.mean(time_array[:, 1]),
    np.mean(time_array[:, 2]),
]

print('Testing...')
test_t0 = time.time()
res = ndcg.Test(net, _predict, True)
print('Evaluation in %.2f seconds' % (time.time() - test_t0))

print('#' * 10, args.dataset, args.u_samp + ' '+ args.i_samp, '#' * 10)
print('Final Epoch:%d, Running time:%.2f, Train:%.2f, Opt:%.2f' % (
    result[2], result[3], result[4], opt_time))
print('Per Epoch Run:%.2f, Per Epoch Train:%.2f, Per Epoch Test:%.2f' % (
    result[5], result[6], result[7]))
print('Ts_auc:%.4f, pre@%d:%.4f, rec@%d:%.4f, ndcg@%d:%.4f' % (
    res['auc'], eval(args.Ks)[-1], res['precision'][-1],
    eval(args.Ks)[-1], res['recall'][-1], eval(args.Ks)[-1], res['ndcg'][-1]))
SAMP_POOL.close()
SAMP_POOL = None
