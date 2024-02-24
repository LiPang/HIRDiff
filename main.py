import argparse
import os
import time

from utility import *
import numpy as np
import torch
import torch as th
import torch.nn.functional as nF
from pathlib import Path
from guided_diffusion import utils
from guided_diffusion.create import create_model_and_diffusion_RS
import scipy.io as sio
from collections import OrderedDict
from os.path import join
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from guided_diffusion.core import imresize, blur_kernel
from math import sqrt, log
import warnings
import matplotlib

def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--baseconfig', type=str, default='configs/base.json',
                        help='JSON file for creating model and diffusion')
    parser.add_argument('-dr', '--dataroot', type=str, default='data')  # dataroot with
    parser.add_argument('-rs', '--resume_state', type=str,
                        default='checkpoints/diffusion/I190000_E97')

    # hyperparameters
    parser.add_argument('-eta1', '--eta1', type=float, default=80)
    parser.add_argument('-eta2', '--eta2', type=float, default=2)
    parser.add_argument('--k', type=float, default=8)
    parser.add_argument('-step', '--step', type=int, default=20)

    # datasets
    parser.add_argument('-dn', '--dataname', type=str, default='WDC',
                        choices=['WDC', 'Houston', 'Salinas'])
    parser.add_argument('--task', type=str, default='denoise',
                        choices=['denoise', 'sr', 'inpainting'])
    parser.add_argument('--task_params', type=str, default='50')

    # settings
    parser.add_argument('--beta_schedule', type=str, default='cosine')
    parser.add_argument('--beta_linear_start', type=float, default=1e-6)
    parser.add_argument('--beta_linear_end', type=float, default=1e-2)
    parser.add_argument('--cosine_s', type=float, default=0)
    parser.add_argument('--no_rrqr', default=False, action='store_true')

    parser.add_argument('-gpu', '--gpu_ids', type=str, default="1")
    parser.add_argument('-seed', '--seed', type=int, default=0)
    parser.add_argument('-bs', '--batch_size', type=int, default=1)
    parser.add_argument('-sn', '--samplenum', type=int, default=1)
    parser.add_argument('-sr', '--savedir', type=str, default='results')



    ## parse configs
    args = parser.parse_args()
    args.eta1 *= 256*64
    args.eta2 *= 8*64
    opt = utils.parse(args)
    opt = utils.dict_to_nonedict(opt)
    return opt


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    opt = parse_args_and_config()

    gpu_ids = opt['gpu_ids']
    if gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        device = th.device("cuda")
        print('export CUDA_VISIBLE_DEVICES=' + gpu_ids)
    else:
        device = th.device("cpu")
        print('use cpu')

    ## create model and diffusion process
    model, diffusion = create_model_and_diffusion_RS(opt)

    ## load model
    load_path = opt['resume_state']
    gen_path = '{}_gen.pth'.format(load_path)
    cks = th.load(gen_path)
    new_cks = OrderedDict()
    for k, v in cks.items():
        newkey = k[11:] if k.startswith('denoise_fn.') else k
        new_cks[newkey] = v
    model.load_state_dict(new_cks, strict=False)
    model.to(device)
    model.eval()

    ## seed
    seeed = opt['seed']
    seed_everywhere(seeed)

    ## params
    param = dict()
    param['task'] = opt['task']
    param['eta1'] = opt['eta1']
    param['eta2'] = opt['eta2']


    if opt['dataname'] == 'Houston':
        if opt['task'] == 'denoise':
            opt['dataroot'] = f'../data/Houston18/test/gauss_{opt["task_params"]}/Houston_channel_cropped.mat'
        if opt['task'] == 'sr':
            opt['dataroot'] = f'../data/Houston18/test/gauss_sr_{opt["task_params"]}/Houston_channel_cropped.mat'
        if opt['task'] == 'inpainting':
            opt['dataroot'] = f'../data/Houston18/test/gauss_inpainting_{opt["task_params"]}/Houston_channel_cropped.mat'

    if opt['dataname'] == 'WDC':
        if opt['task'] == 'denoise':
            opt['dataroot'] = f'../data/WDC/test/gauss_{opt["task_params"]}/wdc_cropped.mat'
        if opt['task'] == 'sr':
            opt['dataroot'] = f'../data/WDC/test/gauss_sr_{opt["task_params"]}/wdc_cropped.mat'
        if opt['task'] == 'inpainting':
            opt['dataroot'] = f'../data/WDC/test/gauss_inpainting_{opt["task_params"]}/wdc_cropped.mat'

    if opt['dataname'] == 'Salinas':
        if opt['task'] == 'denoise':
            opt['dataroot'] = f'../data/Salinas/test/gauss_{opt["task_params"]}/Salinas_channel_cropped.mat'
        if opt['task'] == 'sr':
            opt['dataroot'] = f'../data/Salinas/test/gauss_sr_{opt["task_params"]}/Salinas_channel_cropped.mat'
        if opt['task'] == 'inpainting':
            opt['dataroot'] = f'../data/Salinas/test/gauss_inpainting_{opt["task_params"]}/Salinas_channel_cropped.mat'


    data = sio.loadmat(opt['dataroot'])
    data['input'] = torch.from_numpy(data['input']).permute(2, 0, 1).unsqueeze(0).float().to(device)
    data['gt'] = torch.from_numpy(data['gt']).permute(2, 0, 1).unsqueeze(0).float().to(device)
    Ch, ms = data['gt'].shape[1], data['gt'].shape[2]
    Rr = 3  # spectral dimensironality of subspace
    K = 1
    model_condition = {'input': data['input'], 'gt': data['gt'], 'sigma': data['sigma']}
    if param['task'] == 'inpainting':
        model_condition['mask'] = torch.from_numpy(data['mask'][None]).to(device).permute(0, 3, 1, 2)
        model_condition['transform'] = lambda x: x
    elif param['task'] == 'sr':
        k_s = 9
        sig = sqrt(4 ** 2 / (8 * log(2)))
        scale = data['scale'].item()
        kernel = blur_kernel(k_s, sig)
        kernel = th.from_numpy(kernel).repeat(Ch,1,1,1).to(device)
        blur = partial(nF.conv2d, weight=kernel, padding=int((k_s - 1) / 2), groups=Ch)
        down = partial(imresize, scale=scale)
        model_condition['transform'] = lambda x: down(blur(x))
    else:
        model_condition['transform'] = lambda x: x

    time_start = time.time()
    u, s, v = th.svd(model_condition['input'].reshape(1, Ch, -1).permute(0, 2, 1))
    E = v[..., :, :Rr*K]

    if not opt['no_rrqr']:
        import matlab.engine
        eng = matlab.engine.start_matlab()
        eng.cd(r'matlab')
        res = eng.sRRQR_rank(E[0].cpu().numpy().T, 1.2, 3, nargout=3)
        param['Band'] = th.Tensor(np.sort(list(res[-1][0][:3]))).type(th.int).to(device)-1

    else:
        param['Band'] = th.Tensor([Ch * i // (K * Rr + 1) for i in range(1, K * Rr + 1)]).type(th.int).to(device)
    print(param['Band'])

    denoise_model = None
    denoise_optim = None
    denoised_fn = {
        'denoise_model': denoise_model,
        'denoise_optim': denoise_optim
    }
    step = opt['step']
    dname = opt['dataname']
    for j in range(opt['samplenum']):
        sample, E = diffusion.p_sample_loop(
            model,
            (1, Ch, ms, ms),
            Rr=Rr,
            step=step,
            clip_denoised=True,
            denoised_fn=denoised_fn,
            model_condition=model_condition,
            param=param,
            save_root=None,
            progress=True,
        )
        K = int(len(param['Band']) / Rr)
        sample = (sample + 1) / 2
        im_out = th.matmul(E, sample.reshape(opt['batch_size'], Rr*K, -1)).reshape(opt['batch_size'], Ch, ms, ms)
        im_out = th.clip(im_out, 0, 1)
        time_end = time.time()
        time_cost = time_end - time_start

        psnr_current = np.mean(cal_bwpsnr(im_out, data['gt']))
        if psnr_current < diffusion.best_psnr:
            im_out = diffusion.best_result

        print('best psnr: %.2f, best ssim: %.2f,' %
              (MSIQA(im_out, data['gt'])[0], MSIQA(im_out, data['gt'])[1]))





