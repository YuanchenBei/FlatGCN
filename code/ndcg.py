import time
import torch
import random
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
from multiprocessing import Pool


def init(args, ratio=0.1, batch_size=10240):
    para_dict = pickle.load(open(args.datadir + args.dataset + '_map.pkl', 'rb'))
    global TR_ITEMS, VA_ITEMS, TS_ITEMS, USER_NUM, ITEM_NUM, ITEM_ARRAY, Ks, TS_LBS, TS_NDCG, BATCH_SIZE, SUB_TEST
    print('ndcg Init for %s, %d user %d items' % (args.dataset, para_dict['user_num'], para_dict['item_num']))
    random.seed(args.seed)
    np.random.seed(args.seed)
    TR_ITEMS = para_dict['tr_items']
    TR_ITEMS = {int(k): list(map(int, v)) for k, v in TR_ITEMS.items()}
    VA_ITEMS = para_dict['va_items']
    VA_ITEMS = {int(k): list(map(int, v)) for k, v in VA_ITEMS.items()}
    TS_ITEMS = para_dict['ts_items']
    TS_ITEMS = {int(k): list(map(int, v)) for k, v in TS_ITEMS.items()}
    SUB_TEST = np.random.choice(list(TS_ITEMS.keys()), int(ratio * len(list(TS_ITEMS.keys()))), replace=False)
    USER_NUM = para_dict['user_num']
    ITEM_NUM = para_dict['item_num']
    ITEM_ARRAY = np.array(list(range(ITEM_NUM)))
    TS_LBS = para_dict['ts_lbs']
    TS_NDCG = para_dict['ts_ndcg']
    BATCH_SIZE = batch_size
    if isinstance(args.Ks, str):
        Ks = eval(args.Ks)
    else:
        Ks = args.Ks


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def AUC(all_item_scores, test_data):
    """
        design for a single user
    """
    r_all = np.zeros((ITEM_NUM,))
    r_all[test_data] = 1
    r = r_all[all_item_scores > -(1e9)]
    test_item_scores = all_item_scores[all_item_scores > -(1e9)]
    return roc_auc_score(r, test_item_scores)


def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1. / np.arange(1, k + 1))
    pred_data = pred_data / scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in Ks:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}


def Test(model, pred_func, sub=True, multicore=1, out=False, test_user=None, n_jobs=4):
    """
    Implement in torch,(same as LightGCN pytorch)
    model is the trained model
    pred_func takes (model and lbs as inputs) and outputs the predicted labels
    """
    t0 = time.time()
    max_K = max(Ks)
    results = {'precision': np.zeros(len(Ks)),
               'recall': np.zeros(len(Ks)),
               'ndcg': np.zeros(len(Ks))}
    u_batch = 48
    users_list = []
    rating_list = []
    groundTrue_list = []
    auc_record = []
    # ratings = []
    if test_user is None:
        if sub:
            test_users = SUB_TEST
        else:
            test_users = list(TS_ITEMS.keys())
    total_batch = len(test_users) // u_batch + 1
    for beg in range(0, len(test_users), u_batch):
        batch_users = test_users[beg:min(beg + u_batch, len(test_users))]
        groundTrue = [TS_ITEMS.get(u, []) for u in batch_users]
        rating = np.stack([pred_func(model, np.array(list(zip([u] * len(ITEM_ARRAY), ITEM_ARRAY))))
                           for u in batch_users])
        exclude_index = []
        exclude_items = []
        for i, u in enumerate(batch_users):
            pos_items = TR_ITEMS.get(u, [])
            pos_items.extend(VA_ITEMS.get(u, []))
            exclude_index.extend([i] * len(pos_items))
            exclude_items.extend(pos_items)
        rating[exclude_index, exclude_items] = -(1e10)
        _, rating_K = torch.topk(torch.from_numpy(rating), k=max_K)
        aucs = [
            AUC(rating[i], test_data) for i, test_data in enumerate(groundTrue)
        ]
        auc_record.extend(aucs)
        del rating
        users_list.append(batch_users)
        rating_list.append(rating_K.cpu())
        groundTrue_list.append(groundTrue)
    assert total_batch == len(users_list)
    X = zip(rating_list, groundTrue_list)
    t1 = time.time()
    if multicore:
        with Pool(n_jobs) as pool:
            pre_results = pool.map(test_one_batch, X)
    else:
        pre_results = list(map(test_one_batch, X))
    t2 = time.time()
    for result in pre_results:
        results['recall'] += result['recall']
        results['precision'] += result['precision']
        results['ndcg'] += result['ndcg']
    results['recall'] /= float(len(test_users))
    results['precision'] /= float(len(test_users))
    results['ndcg'] /= float(len(test_users))
    results['auc'] = np.mean(auc_record)
    return results


def Test_gcn(model, pred_func, smps, sub=True, multicore=1, out=False, test_user=None, n_jobs=4):
    """
    Implement in torch,(same as LightGCN pytorch)
    model is the trained model
    pred_func takes (model and lbs as inputs) and outputs the predicted labels
    """
    t0 = time.time()
    max_K = max(Ks)
    results = {'precision': np.zeros(len(Ks)),
               'recall': np.zeros(len(Ks)),
               'ndcg': np.zeros(len(Ks))}
    u_batch = 48
    users_list = []
    rating_list = []
    groundTrue_list = []
    auc_record = []
    # ratings = []
    if test_user is None:
        if sub:
            test_users = SUB_TEST
        else:
            test_users = list(TS_ITEMS.keys())
    total_batch = len(test_users) // u_batch + 1
    for beg in range(0, len(test_users), u_batch):
        batch_users = test_users[beg:min(beg + u_batch, len(test_users))]
        groundTrue = [TS_ITEMS.get(u, []) for u in batch_users]
        rating = np.stack([pred_func(model, smps, np.array(list(zip([u] * len(ITEM_ARRAY), ITEM_ARRAY))))
                           for u in batch_users])
        exclude_index = []
        exclude_items = []
        for i, u in enumerate(batch_users):
            pos_items = TR_ITEMS.get(u, [])
            pos_items.extend(VA_ITEMS.get(u, []))
            exclude_index.extend([i] * len(pos_items))
            exclude_items.extend(pos_items)
        rating[exclude_index, exclude_items] = -(1e10)
        _, rating_K = torch.topk(torch.from_numpy(rating), k=max_K)
        aucs = [
            AUC(rating[i], test_data) for i, test_data in enumerate(groundTrue)
        ]
        auc_record.extend(aucs)
        del rating
        users_list.append(batch_users)
        rating_list.append(rating_K.cpu())
        groundTrue_list.append(groundTrue)
    assert total_batch == len(users_list)
    X = zip(rating_list, groundTrue_list)
    t1 = time.time()
    if multicore:
        with Pool(n_jobs) as pool:
            pre_results = pool.map(test_one_batch, X)
    else:
        pre_results = list(map(test_one_batch, X))
    t2 = time.time()
    for result in pre_results:
        results['recall'] += result['recall']
        results['precision'] += result['precision']
        results['ndcg'] += result['ndcg']
    results['recall'] /= float(len(test_users))
    results['precision'] /= float(len(test_users))
    results['ndcg'] /= float(len(test_users))
    results['auc'] = np.mean(auc_record)
    return results


def _dcg(x):
    # compute dcg_vle
    return x[0] + np.sum(x[1:] / np.log2(np.arange(2, len(x) + 1)))


def _simp_user_ndcg(model, pred_func, user_block):
    def _simp_ndcg(model, lbs):
        # Compute ncdg when the feeding data is one positive vs all negative
        pred = pred_func(model, lbs)
        labels = lbs[:, -1]
        rerank_indices = np.argsort(pred)[::-1]
        rerank_labels = labels[rerank_indices]
        # DCG scores
        dcgs = np.array([_dcg(rerank_labels[:k]) for k in Ks])
        hrs = np.array([np.sum(rerank_labels[:k]) for k in Ks])
        return np.stack([hrs, dcgs])

    # compute the ndcg value of a given user
    return np.mean(np.stack(
        [_simp_ndcg(model, user_block[_]) for _ in range(user_block.shape[0])]), axis=0)


def _auc(model, pred_func, lbs=None):
    if lbs is None:
        lbs = TS_LBS
    pred = pred_func(model, lbs)
    y_true = lbs[:, -1]
    fpr, tpr, thresholds = roc_curve(y_true, pred, pos_label=1)
    return auc(fpr, tpr)


def _auc_gcn(model, pred_func, smps, lbs=None):
    if lbs is None:
        lbs = TS_LBS
    pred = pred_func(model, smps, lbs)
    y_true = lbs[:, -1]
    fpr, tpr, thresholds = roc_curve(y_true, pred, pos_label=1)
    return auc(fpr, tpr)


def _l1out_test(model, pred_func, blocks=None, partial=False):
    if blocks is None:
        blocks = TS_NDCG
    if partial:
        blocks = TS_NDCG[:partial]
    # Leave one out test
    user_scores = np.stack(list(map(lambda x: _simp_user_ndcg(model, pred_func, x), blocks)))
    scores = np.mean(np.stack(user_scores), axis=0)
    d = {}
    d['hr'] = scores[0]
    d['ndcg'] = scores[1]
    d['auc'] = _auc(model, pred_func, TS_LBS)
    return d


def _fast_user_ndcg(pairs):
    def _simp_ndcg(lbs, pred):
        # Compute ncdg when the feeding data is one positive vs all negative
        labels = lbs[:, -1]
        pred = pred.reshape(-1)
        rerank_indices = np.argsort(pred)[::-1]
        rerank_labels = labels[rerank_indices]
        # DCG scores
        dcgs = np.array([_dcg(rerank_labels[:k]) for k in Ks])
        hrs = np.array([np.sum(rerank_labels[:k]) for k in Ks])
        return np.stack([hrs, dcgs])

    # compute the ndcg value of a given user
    return np.mean(np.stack(
        [_simp_ndcg(pairs[0][_], pairs[1][_]) for _ in range(pairs[0].shape[0])]), axis=0)


def batch_predict(model, pred_func, lbs):
    outs = []
    for begin in range(0, lbs.shape[0], BATCH_SIZE):
        end = min(begin + BATCH_SIZE, lbs.shape[0])
        batch_lbs = lbs[begin:end, :]
        outs.append(pred_func(model, batch_lbs))
    out = np.hstack(outs)
    return out


def _fast_ndcg(model, pred_func, blocks=None, partial=False):
    if blocks is None:
        blocks = TS_NDCG
    if partial:
        blocks = TS_NDCG[:partial]
    user_id = []
    for b in blocks:
        user_id.append((b[0][0][0], len(b)))
    b_size = blocks[0].shape[1]
    lbs = np.vstack(blocks).reshape([-1, 3])
    pred_lbs = batch_predict(model, pred_func, lbs).reshape([-1, b_size, 1])
    s = 0
    user_lbs = []
    for _ in user_id:
        user_lbs.append(pred_lbs[s:s + _[1]])
        s += _[1]
    assert s == pred_lbs.shape[0]
    ndcg_list = list(zip(blocks, user_lbs))
    with Pool(5) as pool:
        user_scores = np.stack(list(pool.map(_fast_user_ndcg, ndcg_list)))
    #     user_scores = np.stack(list(map(_fast_user_ndcg, ndcg_list)))
    scores = np.mean(np.stack(user_scores), axis=0)
    d = {}
    d['hr'] = scores[0]
    d['ndcg'] = scores[1]
    d['auc'] = _auc(model, pred_func, TS_LBS)
    return d


def _auc(model, pred_func, lbs=None):
    if lbs is None:
        lbs = TS_LBS
    #     pred = pred_func(model, lbs)
    pred = batch_predict(model, pred_func, lbs)
    y_true = lbs[:, -1]
    fpr, tpr, thresholds = roc_curve(y_true, pred, pos_label=1)
    return auc(fpr, tpr)
