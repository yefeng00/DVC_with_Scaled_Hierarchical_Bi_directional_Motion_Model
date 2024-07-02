'''
© 2019, JamesChan
forked from https://github.com/One-sixth/ms_ssim_pytorch/ssim.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log10
from dataload.type_convert import yuv_to_rgb, rgb_to_yuv
import numpy as np

from scipy import signal
from scipy import ndimage

def psnr_yuv(imgs1, imgs2, data_range=1.):
    imgs1 = imgs1.cpu().numpy().astype(np.float64)
    imgs2 = imgs2.cpu().numpy().astype(np.float64)
    mse = np.mean(np.square(imgs1[0] - imgs2[0]))
    if np.isnan(mse) or np.isinf(mse):
        return -999.9
    if mse > 1e-10:
        psnr = 10 * np.log10(data_range * data_range / mse)
    else:
        psnr = 999.9
    #print(psnr)
    return psnr

def psnr(img1, img2, data_range=1.):
        img_yuv1 = torch.clamp(rgb_to_yuv(img1.float()), 0, 1)
        img_yuv2 = torch.clamp(rgb_to_yuv(img2.float()), 0, 1)
        y1, u1, v1 = img_yuv1.split([1,1,1], dim=1)
        y2, u2, v2 = img_yuv2.split([1,1,1], dim=1)
        # yuv444 to yuv420
        #print(u1.shape)
        if type == 'nearest':
            u1 = u1[:,:,::2,::2]
            u2 = u2[:,:,::2,::2]
            v1 = v1[:,:,::2,::2]
            v2 = v2[:,:,::2,::2]
        else:
            b,_,h,w = u1.shape
            u1 = torch.mean(u1.reshape(b, h//2, 2, w//2, 2), axis=(-1, -3))
            u2 = torch.mean(u2.reshape(b, h//2, 2, w//2, 2), axis=(-1, -3))
            v1 = torch.mean(v1.reshape(b, h//2, 2, w//2, 2), axis=(-1, -3))
            v2 = torch.mean(v2.reshape(b, h//2, 2, w//2, 2), axis=(-1, -3))

        # calculate loss
        Y_loss = torch.mean(torch.square(y1 - y2))
        U_loss = torch.mean(torch.square(u1 - u2))
        V_loss = torch.mean(torch.square(v1 - v2))
        Y_distortion = 10 * torch.log10(1.0 / Y_loss).item()
        U_distortion = 10 * torch.log10(1.0 / U_loss).item()
        V_distortion = 10 * torch.log10(1.0 / V_loss).item()
        psnr = (Y_distortion * 6 + U_distortion + V_distortion) / 8
        return psnr#np.array([Y_distortion, U_distortion, V_distortion])



def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()


def calc_ssim(img1, img2, data_range=255):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * data_range)**2
    C2 = (K2 * data_range)**2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(window, img1 * img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2 * img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1 * img2, mode='valid') - mu1_mu2

    return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                         (sigma1_sq + sigma2_sq + C2)),
            (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))

def get_msssim(img1, img2, data_range=255, use_cuda=True):
    '''
    img1 and img2 are 2D arrays
    '''
    level = 5
    weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    height, width = img1.shape
    if height < 176 or width < 176:
        # according to HM implementation
        level = 4
        weight = np.array([0.0517, 0.3295, 0.3462, 0.2726])
    if height < 88 or width < 88:
        assert False
    im1 = img1.astype(np.float64)
    im2 = img2.astype(np.float64)
    #im1 = img1
    #im2 = img2
    downsample_filter = np.ones((2, 2)) / 4.0
    mssim = np.array([])
    mcs = np.array([])
    for _ in range(level):
        ssim_map, cs_map = calc_ssim(im1, im2, data_range=data_range)
        mssim = np.append(mssim, ssim_map.mean())
        mcs = np.append(mcs, cs_map.mean())
        filtered_im1 = ndimage.filters.convolve(im1, downsample_filter,
                                                mode='reflect')
        filtered_im2 = ndimage.filters.convolve(im2, downsample_filter,
                                                mode='reflect')
        im1 = filtered_im1[::2, ::2]
        im2 = filtered_im2[::2, ::2]
    return (np.prod(mcs[0:level - 1]**weight[0:level - 1]) *
            (mssim[level - 1]**weight[level - 1]))


class MSSSIMLossRGB2YUV(nn.Module):
    r"""Input RGB, calculate MS-SSIM in YUV format.
    """
    def __init__(self, gamma=10):
        super(MSSSIMLossRGB2YUV, self).__init__()
        self.gamma = gamma

    def forward(self, img1, img2):
        r"""
        Args: 
            img1: First RGB Image.
            img2: Second RGB Image.

        Returns:
            distortion_loss : Loss of MS-SSIM in YUV
            Y_distortion    : MS-SSIM in Y
            distortion      : MS-SSIM in YUV
        """
        # rgb to yuv444
        img_yuv1 = torch.clamp(rgb_to_yuv(img1.float()), 0, 1)
        img_yuv2 = torch.clamp(rgb_to_yuv(img2.float()), 0, 1)
        y1, u1, v1 = img_yuv1.split([1,1,1], dim=1)
        y2, u2, v2 = img_yuv2.split([1,1,1], dim=1)
        # yuv444 to yuv420
        u1 = u1[:,:,::2,::2]
        u2 = u2[:,:,::2,::2]
        v1 = v1[:,:,::2,::2]
        v2 = v2[:,:,::2,::2]
        # calculate loss
        #print(y1.shape)
        y1, y2 = y1[0,0].cpu().numpy(), y2[0,0].cpu().numpy()
        u1, u2 = u1[0,0].cpu().numpy(), u2[0,0].cpu().numpy()
        v1, v2 = v1[0,0].cpu().numpy(), v2[0,0].cpu().numpy()
        Y_loss = 1 - get_msssim(y1, y2, 1)
        U_loss = 1 - get_msssim(u1, u2, 1)
        V_loss = 1 - get_msssim(v1, v2, 1)
        Y_distortion = 1 - Y_loss.item()
        U_distortion = 1 - U_loss.item()
        V_distortion = 1 - V_loss.item()
        distortion_loss = 5000 * ((self.gamma * Y_loss + U_loss + V_loss) /
                                  (self.gamma + 2))
        distortion = (6 * Y_distortion + U_distortion + V_distortion) / 8
        return distortion

