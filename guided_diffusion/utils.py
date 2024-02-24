"""
Logger copied from OpenAI baselines to avoid extra RL-based dependencies:
https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/logger.py
"""
import matplotlib.pyplot as plt
import os
import os.path as osp
import json
import datetime
from collections import OrderedDict
import torch
import torch.nn.functional as F
import numpy as np
import math

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def make_beta_schedule(schedule, n_timestep,
                       linear_start=1e-4, linear_end=2e-2,
                       cosine_s=8e-3, k=16):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        # betas = np.linspace(linear_start, linear_end,
        #                     n_timestep, dtype=np.float64)
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            np.arange(n_timestep + 1, dtype=np.float64) /
            n_timestep + cosine_s
        )
        alphas_cumprod = timesteps / (1 + cosine_s)
        alphas_cumprod = np.cos(alphas_cumprod * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        # alphas = timesteps / (1 + cosine_s)
        # alphas = 1/(1+torch.exp((alphas*2-1)*4))
        # alphas = (alphas - alphas.min()) / (alphas.max() - alphas.min())
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = np.clip(betas, a_min=-1, a_max=0.999)
        # betas = betas.clamp(max=0.999)

    elif schedule == "exp":
        # betas = np.linspace(0.15, 0.155, n_timestep+1, dtype=np.float64) ** 2
        # alphas_cumprod = np.cumprod(1 - betas)
        # alphas_cumprod = np.flip(1 - alphas_cumprod)
        alphas_cumprod = np.exp(-k * np.arange(n_timestep + 1, dtype=np.float64) / n_timestep)
        alphas_cumprod = np.flip(1 - alphas_cumprod)
        # alphas_cumprod = alphas_cumprod + np.exp(-k * (n_timestep + 1) / n_timestep)
        # alphas_cumprod = (alphas_cumprod - alphas_cumprod.min()) / (alphas_cumprod.max() - alphas_cumprod.min()) * (1 - 1e-3) + 1e-3
        alphas_cumprod = (alphas_cumprod - alphas_cumprod.min()) / (alphas_cumprod.max() - alphas_cumprod.min())
        # alphas_cumprod = alphas_cumprod * (1 - 0.9) + 0.9
        alphas_cumprod = alphas_cumprod * (1 - 1e-3) + 1e-3
        # alphas_cumprod = np.r_[alphas_cumprod[:-1], [1e-3]]
        # alphas_cumprod = alphas_cumprod * 0.9
        # sigma = 50 / 255 * 2
        # alphas_bar_T = 1 / (1 + 4 * sigma**2)
        # alphas_cumprod = alphas_cumprod * (1 - alphas_bar_T) + alphas_bar_T
        # plt.plot(alphas_cumprod)
        # plt.show()
        alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = 1 - alphas

        # alphas_cumprod = 1 - 1/(1+np.exp(-6*np.linspace(-1, 1, n_timestep+1)))
        # alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
        # betas = 1 - alphas
    elif schedule == "comb":
        timesteps = (
            torch.arange(n_timestep, dtype=torch.float64) /
            (n_timestep) + cosine_s
        )
        alphas_cumprod = timesteps / (1 + cosine_s)
        alphas_cumprod = torch.cos(alphas_cumprod * math.pi / 2).pow(2)
        alphas_cumprod1 = alphas_cumprod / alphas_cumprod[0]
        alphas_cumprod1 = alphas_cumprod1[50:]

        k = 3
        alphas_cumprod = np.exp(-k * np.arange(n_timestep + 1) / (n_timestep))
        alphas_cumprod = np.flip(1 - alphas_cumprod)
        alphas_cumprod = alphas_cumprod + np.exp(-k * (n_timestep + 1) / n_timestep)
        alphas_cumprod = alphas_cumprod[:52]
        alphas_cumprod2 = (alphas_cumprod - alphas_cumprod.min()) / (alphas_cumprod.max() - alphas_cumprod.min()) * 0.5 + 0.5
        alphas_cumprod2 = alphas_cumprod2[:-1]

        alphas_cumprod = np.r_[alphas_cumprod2, alphas_cumprod1]
        alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = 1 - alphas

    else:
        raise NotImplementedError(schedule)
    return betas

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def harr_downsampling(img: torch.Tensor):
    channel_in = img.shape[1]
    haar_weights = torch.ones((4, 1, 2, 2), requires_grad=False)

    haar_weights[1, 0, 0, 1] = -1
    haar_weights[1, 0, 1, 1] = -1

    haar_weights[2, 0, 1, 0] = -1
    haar_weights[2, 0, 1, 1] = -1

    haar_weights[3, 0, 1, 0] = -1
    haar_weights[3, 0, 0, 1] = -1

    haar_weights = torch.cat([haar_weights] * channel_in, 0).to(img.device)
    out = F.conv2d(img, haar_weights, bias=None, stride=2, groups=channel_in) / 4.0
    out = out.reshape([img.shape[0], channel_in, 4, img.shape[2] // 2, img.shape[3] // 2])
    out = torch.transpose(out, 1, 2)

    out = out.reshape([img.shape[0], channel_in * 4, img.shape[2] // 2, img.shape[3] // 2])
    return out

def mkdirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)

def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')

def parse(args):
    args = vars(args)
    opt_path = args['baseconfig']
    # remove comments starting with '//'
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    for key in args:
        if args[key] is not None:
            opt[key] = args[key]
    
    return opt

class NoneDict(dict):
    def __missing__(self, key):
        return None

def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt

def svd_denoise(inp_img, rank):
    Bb, Cc, H, W = inp_img.shape
    u, s, v = torch.svd(inp_img.reshape(Bb, Cc, -1).permute(0, 2, 1))
    A = v[:, :, :rank]
    M = u[:, :, :rank] @ torch.diag_embed(s[:, :rank])
    M = M.permute(0, 2, 1).reshape(Bb, -1, H, W)
    x_denoised = torch.einsum('bcr, brhw->bchw', A, M)
    return x_denoised, M, A


def diff_2d(img):
    # _, _, H, W = img.shape
    diff_x, diff_y = torch.zeros_like(img).to(img.device), torch.zeros_like(img).to(img.device)
    diff_x[..., :-1, :] = torch.diff(img, dim=-2)
    diff_x[..., -1, :] = img[..., 0, :] - img[..., -1, :]
    diff_y[..., :-1] = torch.diff(img, dim=-1)
    diff_y[..., -1] = img[..., 0] - img[..., -1]
    return diff_x, diff_y

def diff_3d(img, keepdim=True):
    # _, _, H, W = img.shape
    if keepdim:
        diff_x, diff_y, diff_z = torch.zeros_like(img).to(img.device), torch.zeros_like(img).to(img.device), torch.zeros_like(img).to(img.device)
        diff_x[..., :-1, :] = torch.diff(img, dim=-2)
        diff_x[..., -1, :] = img[..., 0, :] - img[..., -1, :]
        diff_y[..., :-1] = torch.diff(img, dim=-1)
        diff_y[..., -1] = img[..., 0] - img[..., -1]
        diff_z[:, :-1, ...] = torch.diff(img, dim=-3)
        diff_z[:, -1, ...] = img[:, 0, ...] - img[:, -1, ...]
    else:
        diff_x = torch.diff(img, dim=-2)
        diff_y = torch.diff(img, dim=-1)
        diff_z = torch.diff(img, dim=-3)
    return diff_x, diff_y, diff_z


def img2patch(img, ps=5, stride=1):
    kernel = torch.eye(ps ** 2, device=img.device).reshape(ps, ps, ps ** 2)
    kernel = kernel.permute(2, 0, 1).unsqueeze(1)
    img_patch = []
    for i in range(img.shape[-1]):
        temp = img[..., i][None, None, ...].float()
        temp = torch.nn.functional.conv2d(temp, kernel, stride=stride, padding=0)
        temp = temp[0]
        img_patch.append(temp)
    img_patch = torch.cat(img_patch, dim=0)
    img_patch = img_patch.reshape(img_patch.shape[0], -1)
    return img_patch

def neighbor_search(E_img_average_patch, Hp, Wp, sw, pn, step):
    img_idx = torch.arange(Hp*Wp).reshape(Hp, Wp).to(E_img_average_patch.device)
    grid_x = list(range(0, Hp, step)) + list(range(Hp-step, Hp, 1))
    grid_y = list(range(0, Wp, step)) + list(range(Wp-step, Wp, 1))
    neighbor_index = torch.zeros((pn, len(grid_x)*len(grid_y)), dtype=torch.int64, device=E_img_average_patch.device)
    neighbor_dist = torch.zeros((pn, len(grid_x) * len(grid_y)), dtype=torch.float32, device=E_img_average_patch.device)
    weight_nb = torch.zeros((pn, len(grid_x) * len(grid_y)), device=E_img_average_patch.device)
    k = 0
    for i in grid_x:
        for j in grid_y:
            # search neighborhood
            window_idx = img_idx[max(i - sw, 0):min(i + sw + 1, Hp),
                         max(j - sw, 0):min(j + sw + 1, Wp)].flatten()

            window = E_img_average_patch[:, window_idx]
            dist = (window - E_img_average_patch[:, img_idx[i, j]][:, None])**2
            dist = dist.sum(0)
            idx = dist.argsort()[:pn]
            neighbor_dist[:, k] = dist[idx]
            neighbor_index[:, k] = window_idx[idx]
            weight_nb[:, k] = torch.softmax(-100 * neighbor_dist[:, k], dim=0)
            k += 1
    return neighbor_index, neighbor_dist, weight_nb

def subspace_denoising(E_img, E_img_patch, neighbor_index, neighbor_dist, weight_nb, ps, Hp, Wp):
    # img_patch_restored = torch.zeros_like(E_img_patch, device=E_img.device)
    # weight_patch = torch.zeros_like(E_img_patch, device=E_img.device)
    # neighbors = [E_img_patch[:, neighbor_index[:, t]][None]
    #              for t in range(neighbor_index.shape[-1])]
    # neighbors = torch.cat(neighbors, dim=0)
    # for i in range(neighbors.shape[0]):
    #     window_img_restored = neighbors[i]
    #     img_patch_restored[:, neighbor_index[:, i]] += window_img_restored
    #     weight_patch[:, neighbor_index[:, i]] += 1

    img_patch_restored = torch.zeros_like(E_img_patch, device=E_img.device)
    weight_patch = torch.zeros_like(E_img_patch, device=E_img.device)
    neighbors = [E_img_patch[:, neighbor_index[:, t]][None]
                 for t in range(neighbor_index.shape[-1])]
    neighbors = torch.cat(neighbors, dim=0)

    img_patch_restored_nb = (neighbors @ weight_nb.T.unsqueeze(-1)).squeeze()
    img_patch_restored[:, neighbor_index[0, :]] = img_patch_restored_nb.T
    weight_patch[:, neighbor_index[0, :]] += 1

    # img_patch_restored = E_img_patch
    # weight_patch += 1

    # patch to img
    img_restored = torch.zeros_like(E_img, device=E_img.device)
    weight_restored = torch.zeros_like(E_img, device=E_img.device)
    k = 0
    for o in range(E_img.shape[-1]):
        for i in range(ps):
            for j in range(ps):
                img_restored[i:i+Hp, j:j+Wp, o] += img_patch_restored[k, :].reshape(Hp, Wp)
                weight_restored[i:i+Hp, j:j+Wp, o] += weight_patch[k, :].reshape(Hp, Wp)
                # weight_restored[i:i + Hp, j:j + Wp, o] += np_reshape(weight_patch[k, :], (Hp, Wp)).T
                k += 1
    img_restored = img_restored / weight_restored
    return img_restored
