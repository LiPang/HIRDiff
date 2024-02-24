import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from functools import partial
import torch
import imgvision as iv

class Bandwise(object):
    def __init__(self, index_fn):
        self.index_fn = index_fn

    def __call__(self, X, Y):
        C = X.shape[-3]
        bwindex = []
        for ch in range(C):
            x = torch.squeeze(X[...,ch,:,:].data).cpu().numpy()
            y = torch.squeeze(Y[...,ch,:,:].data).cpu().numpy()
            index = self.index_fn(x, y)
            bwindex.append(index)
        return bwindex

cal_bwssim = Bandwise(partial(compare_ssim, data_range=1))
cal_bwpsnr = Bandwise(partial(compare_psnr, data_range=1))


def cal_sam(X, Y, eps=1e-8):
    X = torch.squeeze(X.data).cpu().numpy()
    Y = torch.squeeze(Y.data).cpu().numpy()
    tmp = np.sum(X*Y, axis=0) / (np.sqrt(np.sum(X**2, axis=0)) * np.sqrt(np.sum(Y**2, axis=0)) + eps)
    return np.mean(np.real(np.arccos(tmp)))

def cal_ergas(X, Y):
    if len(X.shape) == 4:
        X = X[None, ...]
        Y = Y[None, ...]
    # Metric = iv.spectra_metric(Y[0, 0, ...].permute(1,2,0).detach().cpu().numpy(), X[0, 0, ...].permute(1,2,0).detach().cpu().numpy(),
    #                            scale=1)
    # ERGAS = Metric.ERGAS()

    ergas = 0
    for i in range(X.size(2)):
        ergas = ergas + torch.nn.functional.mse_loss(X[:,:, i, ...], Y[:,:, i, ...]) / (torch.mean(X[:,:, i, ...])+1e-6) #** 2
    ergas = 100 * torch.sqrt(ergas / X.size(2))
    ergas = ergas.item()
    return ergas

def MSIQA(X, Y):
    psnr = np.mean(cal_bwpsnr(X, Y))
    ssim = np.mean(cal_bwssim(X, Y))
    sam = cal_sam(X, Y)
    ergas = cal_ergas(X, Y)
    return psnr, ssim, sam, ergas

if __name__ == '__main__':
    from scipy.io import loadmat
    import torch
    hsi = loadmat('/home/cxy/LPang/GlowModels/data/CAVE/val/gauss_30/balloons_ms.mat')['gt']
    R_hsi = loadmat('/home/cxy/LPang/GlowModels/matlab/Result/gauss_50/balloons_ms/BM4D.mat')['R_hsi']
    print(MSIQA(torch.from_numpy(hsi), torch.from_numpy(R_hsi)))
