import random
import numpy as np
import torch
from matplotlib import pyplot as plt

def my_diff(x):
    diff_1, diff_2, diff_3 = torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)
    diff_1[:, :-1, ...] = x[:, :-1, ...] - x[:, 1:, ...]
    diff_1[:, -1, ...] = x[:, -1, ...] - x[:, 0, ...]
    diff_2[:, :, :-1, ...] = x[:, :, :-1, ...] - x[:, :, 1:, ...]
    diff_2[:, :, -1, ...] = x[:, :, -1, :] - x[:, :, 0, ...]
    diff_3[..., :-1] = x[..., :-1] - x[..., 1:]
    diff_3[..., -1] = x[..., -1] - x[..., 0]
    return diff_1, diff_2, diff_3

def my_svd(x, rank):
    B, C, H, W = x.shape
    u, s, v = torch.svd(x.reshape(B, C, -1).permute(0, 2, 1))
    A = v[:, :, :rank]
    M = u[:, :, :rank] @ torch.diag_embed(s[:, :rank])
    M = M.permute(0, 2, 1).reshape(B, -1, H, W)
    x_lr = torch.einsum('bcr, brhw->bchw', A, M)
    return x_lr, A, M


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def minmax_normalize(array):
    array = array.astype(np.float)
    amin = np.min(array)
    amax = np.max(array)
    return (array - amin) / (amax - amin)

def torch2numpy(hsi, use_2dconv):
    if use_2dconv:
        R_hsi = hsi.data[0].cpu().numpy().transpose((1, 2, 0))
    else:
        R_hsi = hsi.data[0].cpu().numpy()[0, ...].transpose((1, 2, 0))
    return R_hsi





def seed_everywhere(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.