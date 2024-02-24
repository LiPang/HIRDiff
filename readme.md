# HIR-Diff: Unsupervised Hyperspectral Image Restoration Via Improved Diffusion Models

---

This is the Pytorch implementation of Unsupervised Hyperspectral Image Restoration Via Improved Diffusion Models.

<div align=left>
<img src="imgs/diffusion.png" height="100%" width="90%"/>
</div>

## Introduction
Hyperspectral image (HSI) restoration aims at recovering clean images from degraded observations and plays a vital role in downstream tasks. Existing model-based methods have limitations in accurately modeling the complex image characteristics with handcraft priors, and deep learning-based methods suffer from poor generalization ability. To alleviate these issues, this paper proposes an unsupervised HSI restoration framework with pre-trained diffusion model (HIR-Diff), which restores the clean HSIs from the product of two low-rank components, i.e., the reduced image and the coefficient matrix. Specifically, the reduced image, which has a low spectral dimension, lies in the image field and can be inferred from our improved diffusion model where a new guidance function with total variation (TV) prior is designed to ensure that the reduced image can be well sampled. The coefficient matrix can be effectively pre-estimated based on singular value decomposition (SVD) and rank-revealing QR (RRQR) factorization. Furthermore, a novel exponential noise schedule is proposed to accelerate the restoration process (about 5$\times$ acceleration for denoising) with little performance decrease. Extensive experimental results validate the superiority of our method in both performance and speed on a variety of HSI restoration tasks, including HSI denoising, noisy HSI super-resolution, and noisy HSI inpainting.

## Hightlights
<div align=center>
<img src="imgs/result_light.png" height="100%" width="100%"/>
</div>

## Environment

---
```bash
conda create -n hirdiff python=3.9
```
```bash
conda activate hirdiff
```
```bash
pip3 install -r requirement.txt
```


## HSI Restoration

---

### Download the pretrained diffusion model
downloading the pretrained diffusion model I190000_E97_gen.pth from  
[https://github.com/wgcban/ddpm-cd](https://github.com/wgcban/ddpm-cd)
[https://www.dropbox.com/sh/z6k5ixlhkpwgzt5/AAApBOGEUhHa4qZon0MxUfmua?dl=0](https://www.dropbox.com/sh/z6k5ixlhkpwgzt5/AAApBOGEUhHa4qZon0MxUfmua?dl=0)
and put the model into checkpoints\diffusion

### Testing

Denoising on Houston dataset (sigma=50) (psnr: 36.01)
```bash
python main.py -eta1 16 -eta2 10 --k 8 -step 20 -dn Houston --task denoise --task_params 50 -gpu 0   
```

Super-Resolution on WDC dataset (upscale factor=x4) (psnr: 34.68)
```bash
python main.py -eta1 500 -eta2 12 --k 8 -step 20 -dn WDC --task sr --task_params 0.25 -gpu 0  
```

Inpainting on Salinas dataset (masking rate=0.8) (psnr: )
```bash
python main.py -eta1 8 -eta2 6 --k 5 -step 20 -dn Salinas --task inpainting --task_params 0.8 -gpu 0  
```

### The effectiveness of RRQR
To disable RRQR, adding '--no_rrqr' and the bands are selected at equal intervals
```bash
python main.py -eta1 16 -eta2 10 --k 8 -step 20 -dn Houston --task denoise --task_params 50 -gpu 0 --no_rrqr
python main.py -eta1 500 -eta2 12 --k 8 -step 20 -dn WDC --task sr --task_params 0.25 -gpu 0 --no_rrqr
python main.py -eta1 8 -eta2 6 --k 5 -step 20 -dn Salinas --task inpainting --task_params 0.8 -gpu 0 --no_rrqr
```

### Comparison with other schedules

linear schedule (psnr: 34.33)
```bash
python main.py -eta1 20 -eta2 1 -step 20 -dn Houston --task denoise --task_params 50 -gpu 0 --beta_schedule linear      
```
cosine schedule (psnr: 34.61)
```bash
python main.py -eta1 20 -eta2 1 -step 20 -dn Houston --task denoise --task_params 50 -gpu 0 --beta_schedule cosine      
```
exponential schedule (psnr: 36.01)
```bash
python main.py -eta1 16 -eta2 10 --k 8 -step 20 -dn Houston --task denoise --task_params 50 -gpu 0 --beta_schedule exp   
```

