import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

def build_index(dataset_name):

    ui_mat = np.loadtxt('data/%s.txt' % dataset_name, dtype=np.int32)

    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()

    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])

    return u2i_index, i2u_index

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):

        # uid = np.random.randint(1, usernum + 1)
        while len(user_train[uid]) <= 1: uid = np.random.randint(1, usernum + 1)

        seq_items = np.zeros([maxlen], dtype=np.int32)

        seq_actions = np.zeros([maxlen], dtype=np.int32) #NEW

        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)

        nxt = user_train[uid][-1][0]
        
        idx = maxlen - 1

        ts = set([i[0] for i in user_train[uid]])

        for i in reversed(user_train[uid][:-1]):
            seq_items[idx] = i[0]
            seq_actions[idx] = i[1] #NEW
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i[0]
            idx -= 1
            if idx == -1: break

        return (uid, [seq_items, seq_actions],  pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i, t = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        t = int(t) #NEW
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append((i, t)) #NEW

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

# TODO: merge evaluate functions for test and val set
# evaluate on test set

def evaluate_model(model, dataset, args, mode='test'):
    """
    Evaluate SASRec model on validation or test set.
    
    Args:
        model: Trained SASRec model with `predict` method.
        dataset: [train, valid, test, usernum, itemnum]
        args: Hyperparameters with args.maxlen
        mode: 'valid' or 'test'
    
    Returns:
        Tuple: (NDCG@10, HR@10, MRR)
    """
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG, HT, MRR = 0.0, 0.0, 0.0
    NDCG_5, HT_5 = 0.0, 0.0
    HR_1 = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    for u in users:
        if len(train[u]) < 1: continue
        if mode == 'valid' and len(valid[u]) < 1: continue
        if mode == 'test' and len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1

        # If evaluating on test, include valid[0] as part of input
        if mode == 'test':
            seq[idx] = valid[u][0][0]
            idx -= 1

        for i in reversed(train[u]):
            seq[idx] = i[0]  # i[0] is item_id
            idx -= 1
            if idx == -1: break

        rated = set(i[0] for i in train[u])
        rated.add(0)

        # Set target item
        if mode == 'valid':
            target_item = valid[u][0][0]
        else:
            target_item = test[u][0][0]

        item_idx = [target_item]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1
        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        
        if rank < 5:
            NDCG_5 += 1 / np.log2(rank + 2)
            HT_5 += 1

        if rank == 1:
            HR_1 += 1
        #MRR
        MRR += 1.0 / (rank + 1)

        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return_vals = { 
        'NDCG@10': NDCG / valid_user,
        'NDCG@5': NDCG_5 / valid_user,
        'HR@10': HT / valid_user,
        'HR@5': HT_5 / valid_user,
        'MRR': MRR / valid_user,
        'HR@1': HR_1 / valid_user
    }

    return return_vals


def evaluate_by_seq_length(model, dataset, args, X=5, mode='test'):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    short_metrics = {'NDCG': 0.0, 'HR': 0.0, 'MRR': 0.0, 'count': 0}
    long_metrics  = {'NDCG': 0.0, 'HR': 0.0, 'MRR': 0.0, 'count': 0}

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    for u in users:
        if len(train[u]) < 1: continue
        if mode == 'valid' and len(valid[u]) < 1: continue
        if mode == 'test' and len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1

        # Include valid item in test sequences
        if mode == 'test':
            seq[idx] = valid[u][0]
            idx -= 1

        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        true_seq_len = np.count_nonzero(seq)
        rated = set(train[u])
        rated.add(0)

        # Define the target item
        target_item = valid[u][0] if mode == 'valid' else test[u][0]

        item_idx = [target_item]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]
        rank = predictions.argsort().argsort()[0].item()

        if true_seq_len < X:
            metrics = short_metrics
        else:
            metrics = long_metrics

        metrics['NDCG'] += 1 / np.log2(rank + 2) if rank < 10 else 0
        metrics['HR'] += 1 if rank < 10 else 0
        metrics['MRR'] += 1.0 / (rank + 1)
        metrics['count'] += 1

    def average(m):
        if m['count'] == 0:
            return {'NDCG@10': 0, 'HR@10': 0, 'MRR': 0}
        return {
            'NDCG@10': m['NDCG'] / m['count'],
            'HR@10': m['HR'] / m['count'],
            'MRR': m['MRR'] / m['count']
        }

    return {
        'short (<{})'.format(X): average(short_metrics),
        'long (≥{})'.format(X): average(long_metrics)
    }

def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
