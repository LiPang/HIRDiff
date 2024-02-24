"""
This code started out as a PyTorch port of the following:
https://github.com/HJ-harry/MCG_diffusion/blob/main/guided_diffusion/gaussian_diffusion.py
The conditions are changed and coefficient matrix estimation is added.
"""
import matplotlib.pyplot as plt
import enum
import math
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import numpy as np
import torch as th
from torch.autograd import grad
import torch.nn.functional as nF
from functools import partial
import torch.nn.parameter as Para
from .utils import *
from os.path import join as join
from utility import *
from skimage.restoration import denoise_nl_means, estimate_sigma



class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL

class Param(th.nn.Module):
    def __init__(self, data):
        super(Param, self).__init__()
        self.E = Para.Parameter(data=data)
    
    def forward(self,):
        return self.E

class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    """

    def __init__(
        self,
        *,
        betas
    ):

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        self.num_timesteps = int(betas.shape[0])
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod)
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., self.alphas_cumprod))



    def p_sample_loop(
        self,
        model,
        shape,
        Rr,
        step=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_condition=None,
        param=None,
        save_root=None,
        progress=True
    ):
        finalX = None
        finalE = None

        for (sample, E) in self.p_sample_loop_progressive(
            model,
            shape,
            Rr,
            step=step,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_condition=model_condition,
            param=param,
            save_root=save_root
        ):
            finalX = sample
            finalE = E
             
        return finalX["sample"], finalE
        # return finalX["pred_xstart"], finalE

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        Rr,
        step=None,
        noise=None,
        clip_denoised=True,
        model_condition=None,
        device=None,
        param=None,
        denoised_fn=None,
        save_root=None   # use it for output intermediate predictions
        ):
        Bb, Cc, Hh, Ww = shape
        Rr = Rr
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn((Bb, Rr, Hh, Ww), device=device) # 初始大小是[B, r, H, W]

        if step is None:
            step = self.num_timesteps
        indices = list(np.arange(0, self.num_timesteps, self.num_timesteps//step))[::-1]
        indices_next = indices[1:] + [-1]
        from tqdm import tqdm
        pbar = tqdm(enumerate(zip(indices, indices_next)), total=len(indices))

        # coefficient matrix estimation: Eq.(14)
        u, s, v = th.svd(model_condition['input'].reshape(Bb, Cc, -1).permute(0, 2, 1))
        v[:, :, 1] *= -1
        E = v[..., :, :3]
        coef = th.inverse(th.index_select(E, 1, param['Band']))
        E = E @ coef

        self.best_result, self.best_psnr = None, 0
        norm_list, psnr_list, result_list = [], [], []
        alphas_bar_list = []
        for iteration, (i, j) in pbar:
            t = th.tensor([i] * shape[0], device=device)
            t_next = th.tensor([j] * shape[0], device=device)
            # re-instantiate requires_grad for backpropagation
            img = img.requires_grad_()

            x, eps = img, 1e-9
            B = x.shape[0]

            alphas_bar = th.FloatTensor([self.alphas_cumprod_prev[int(t.item()) + 1]]).repeat(B, 1).to(x.device)
            alphas_bar_next = th.FloatTensor([self.alphas_cumprod_prev[int(t_next.item()) + 1]]).repeat(B, 1).to(
                x.device)

            # DDIM: Algorithm 1 in the paper
            model_output = model(x, alphas_bar)
            pred_xstart = (x - model_output * (1 - alphas_bar).sqrt()) / alphas_bar.sqrt()
            if clip_denoised:
                pred_xstart = pred_xstart.clamp(-1, 1)

            # update
            xhat = (pred_xstart + 1) / 2
            xhat = th.matmul(E, xhat.reshape(Bb, Rr, -1)).reshape(*shape)

            # parameters
            eta = 0
            c1 = (
                    eta * (
                    (1 - alphas_bar / alphas_bar_next) * (1 - alphas_bar_next) / (1 - alphas_bar + eps)).sqrt()
            )
            c2 = ((1 - alphas_bar_next) - c1 ** 2).sqrt()
            xt_next = alphas_bar_next.sqrt() * pred_xstart + c1 * th.randn_like(x) + c2 * model_output

            param['iteration'] = iteration
            if param['task'] == 'sr':
                loss_condition = self.loss_sr(param, model_condition, xhat)
            elif param['task'] == 'denoise':
                loss_condition = self.loss_denoise(param, model_condition, xhat)
            elif param['task'] == 'inpainting':
                loss_condition = self.loss_inpainting(param, model_condition, xhat)
            else:
                raise ValueError('invalid task name')

            norm_gradX = grad(outputs=loss_condition, inputs=img)[0]
            xt_next = xt_next - norm_gradX
            del norm_gradX


            out = {"sample": xt_next, "pred_xstart": pred_xstart}
            yield out, E
            img = out["sample"]
            # Clears out small amount of gpu memory. If not used, memory usage will accumulate and OOM will occur.
            img.detach_()

            # evaluate
            alphas_bar_list.append(alphas_bar.item())
            norm_list.append(loss_condition.item())
            psnr_current = np.mean(cal_bwpsnr(xhat, model_condition['gt']))
            if psnr_current > self.best_psnr:
                self.best_psnr = psnr_current
                self.best_result = xhat.clone()

            psnr_list.append(psnr_current)
            pbar.set_description("%d/%d, psnr: %.2f" % (iteration, len(indices), psnr_list[-1]))


        # plt.plot(alphas_bar_list, c='r', label='alphas_bar')
        # plt.legend()
        # plt.show()

        plt.plot(psnr_list)
        plt.ylabel('PSNR')
        plt.show()

        # plt.plot(norm_list)
        # plt.ylabel('guidance function loss')
        # plt.show()


    def loss_sr(self, param, model_condition, xhat):
        input = model_condition['input']
        weight = 1
        loss_1 = param['eta1'] * (th.norm(weight * (input -
                    model_condition['transform'](xhat)), p=2)) ** 2 / xhat.numel()

        # regularization term
        weight_dx, weight_dy, weight_dz = 1, 1, 1
        xhat_dx, xhat_dy, xhat_dz = diff_3d(xhat)
        loss_2 = (param['eta2'] * th.norm(weight_dx * xhat_dx, p=1) +
                  param['eta2'] * th.norm(weight_dy * xhat_dy, p=1) +
                  param['eta2'] * th.norm(weight_dz * xhat_dz, p=1)
                  ) / xhat.numel()


        loss_condition = loss_1 + loss_2
        return loss_condition

    def loss_inpainting(self, param, model_condition, xhat):
        input = model_condition['input']

        # fidelity term
        weight = model_condition['mask']
        frac = weight.sum()
        loss_1 = param['eta1'] * (
            th.norm(weight * (input - model_condition['transform'](xhat)), p=2)) ** 2 / frac

        weight_dx, weight_dy, weight_dz = 1, 1, 1
        xhat_dx, xhat_dy, xhat_dz = diff_3d(xhat, keepdim=False)
        norm_rank = 1
        loss_2 = (param['eta2'] * th.norm(weight_dx * xhat_dx, p=norm_rank) **norm_rank / xhat_dx.numel() +
                  param['eta2'] * th.norm(weight_dy * xhat_dy, p=norm_rank) **norm_rank / xhat_dy.numel() +
                  param['eta2'] * th.norm(weight_dz * xhat_dz, p=norm_rank) **norm_rank / xhat_dz.numel()
                  )

        loss_condition = loss_1 + loss_2
        return loss_condition

    def loss_denoise(self, param, model_condition, xhat):
        input = model_condition['input']

        # fidelity term
        weight = 1
        loss_1 = param['eta1'] * (th.norm(weight * (input - model_condition['transform'](xhat)), p=2)) ** 2 / xhat.numel()

        # regularization term
        x = model_condition['input']
        rank, delta = 3, 0.5
        x_denoised, _, _ = svd_denoise(x, rank)
        x_denoised_dx, x_denoised_dy, x_denoised_dz = diff_3d(x_denoised)
        x_denoised_dx = x_denoised_dx.mean(1, keepdim=True).abs()
        x_denoised_dy = x_denoised_dy.mean(1, keepdim=True).abs()
        x_denoised_dz = x_denoised_dz.mean(1, keepdim=True).abs()

        weight_dx = x_denoised_dx.max() / (x_denoised_dx + delta * x_denoised_dx.max())
        weight_dy = x_denoised_dy.max() / (x_denoised_dy + delta * x_denoised_dy.max())
        weight_dz = x_denoised_dz.max() / (x_denoised_dz + delta * x_denoised_dz.max())

        max_value = max(weight_dx.max(), weight_dy.max(), weight_dz.max())
        weight_dx, weight_dy, weight_dz = weight_dx / max_value, weight_dy / max_value, weight_dz / max_value

        xhat_dx, xhat_dy, xhat_dz = diff_3d(xhat)
        loss_2 = (param['eta2'] * th.norm(weight_dx * xhat_dx, p=1) +
                  param['eta2'] * th.norm(weight_dy * xhat_dy, p=1) +
                  param['eta2'] * th.norm(weight_dz * xhat_dz, p=1)
                  ) / xhat.numel()


        loss_condition = loss_1 + loss_2
        return loss_condition



    def p_sample(
        self,
        model,
        x,
        t,
        t_next=None,
        clip_denoised=True,
        denoised_fn=None,
        eps=1e-9,
    ):
        # predict x_start
        B = x.shape[0]
        noise_level = th.FloatTensor([self.sqrt_alphas_cumprod_prev[int(t.item()) + 1]]).repeat(B, 1).to(x.device)
        noise_level_next = th.FloatTensor([self.sqrt_alphas_cumprod_prev[int(t_next.item()) + 1]]).repeat(B, 1).to(x.device)
        model_output = model(x, noise_level)
        pred_xstart = (x - model_output * (1 - noise_level).sqrt()) / noise_level.sqrt()
        if clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1)


        # next step
        eta = 0
        c1 = (
                eta * ((1 - noise_level / noise_level_next) * (1 - noise_level_next) / (1 - noise_level + eps)).sqrt()
        )
        c2 = ((1 - noise_level_next) - c1 ** 2).sqrt()
        xt_next = noise_level_next.sqrt() * pred_xstart + c1 * th.randn_like(x) + c2 * model_output
        out = {"sample": xt_next, "pred_xstart": pred_xstart}
        return out


