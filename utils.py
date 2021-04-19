import os
import numpy as np
import random
import torch


def set_global_seeds(i=2021):
    np.random.seed(i)
    random.seed(i)
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)


def check_path(fpath):
    return os.path.exists(fpath)


def batch_generator(data, batch_size, shuffle=False, max_len=1024, device='cpu'):
    size = len(data)
    indices = list(range(size))
    if shuffle:
        random.shuffle(indices)

    for i in range(0, size, batch_size):
        batch = [[], [], [], []]    #pre_seq, post_seq, level_seq
        for j in indices[i:i+batch_size]:
            row = data.iloc[j]
            batch[0].append(torch.LongTensor(row['pre'][:max_len]).to(device))
            batch[1].append(torch.LongTensor(row['post'][:max_len]).to(device))
            batch[2].append(torch.LongTensor(row['level'][:max_len]).to(device))
            batch[3].append(row['label']-1)
        batch[3] = torch.LongTensor(batch[3]).to(device)
        yield batch
  

def test():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.nn.utils.rnn as rnn_utils

    embed = nn.Embedding(10, 3)
    rnn = nn.GRU(3, 2, batch_first=True, bidirectional=True)

    batch = [[1,2,3], [4,5,6,7], [3], [2,1]]
    batch = [torch.LongTensor(ele) for ele in batch]
    pad_batch = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0)
    print(pad_batch.shape)
    emb = embed(pad_batch)

    pack_inp = rnn_utils.pack_padded_sequence(emb, lengths=[3,4,1,2], batch_first=True, enforce_sorted=False)
    out, hidden = rnn(pack_inp)
    print(hidden.shape)
    print(type(out))
    out, real_len = rnn_utils.pad_packed_sequence(out, batch_first=True, padding_value=float('-inf'))
    print(real_len)
    print(out.shape)  # [batch_size, max_len, hiiden*2]
    print(out)

    out = torch.transpose(out, 1, 2)
    out = F.max_pool1d(out, out.size(2)).squeeze(2) # [batch_size, hidden_dim*2]
    print(out.shape)
    print(out)

if __name__ == '__main__':
    test()