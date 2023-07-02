from ast import Not
import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import cv2 as cv
import torch.nn as nn
from collections import OrderedDict
from torchvision import transforms
import copy
from PIL import Image
from tqdm import tqdm
import random
from scipy.fftpack import dct, idct
import imageio
import kornia


# Class for CTRL

def dct_fft_impl(v):
    return torch.view_as_real(torch.fft.fft(v, dim=1))  

def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = dct_fft_impl(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V

def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)

def idct_irfft_impl(V):
    return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)

def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)

def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_2d(dct_2d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)

def DCT(x, window_size=32):
    x_dct = torch.zeros_like(x)
    for ch in range(x.shape[0]):
        for w in range(0, x.shape[2], window_size):
            for h in range(0, x.shape[1], window_size):
                sub_dct = dct_2d(x[ch][w:w + window_size, h:h + window_size], norm='ortho')
                x_dct[ch][w:w + window_size, h:h + window_size] = sub_dct
    return x_dct

def IDCT(x, window_size=32):
    x_idct = torch.zeros_like(x)
    for ch in range(x.shape[0]):
        for w in range(0, x.shape[2], window_size):
            for h in range(0, x.shape[1], window_size):
                sub_idct = idct_2d(x[ch][w:w + window_size, h:h + window_size], norm='ortho')
                x_idct[ch][w:w + window_size, h:h + window_size] = sub_idct
    return x_idct

def ctrl_img(x, magnitude = 100, window_size=32, channel_list = [1, 2], pos_list = [15, 31], clip_range = (-1,1)):
    x  = x * 255.
    x = kornia.color.rgb_to_yuv(x)
    x = DCT(x, window_size=window_size)
    position_list = [(pos_list[0], pos_list[0]), (pos_list[1], pos_list[1])]
    
    for ch in channel_list:
        for w in range(0, x.shape[1], window_size):
            for h in range(0, x.shape[2], window_size):
                    for pos in position_list:
                            x[ch, w+pos[0], h+pos[1]] = x[ch, w+pos[0], h+pos[1]] + magnitude
    
    x = IDCT(x)
    
    x = kornia.color.yuv_to_rgb(x)
    
    x /= 255.
    x = torch.clamp(x, min=clip_range[0], max=clip_range[1])
    
    return x

class CTRL():
    def __init__(self, magnitude = 100, size = 32, clip_range = (-1,1)):
        self.magnitude = magnitude
        self.size = size
        self.clip_range = clip_range
        self.totensor = transforms.ToTensor()
    
    def img_poi(self, img):
        img = ctrl_img(self.totensor(img), magnitude = self.magnitude, window_size = self.size, clip_range = self.clip_range).permute(1, 2, 0).numpy()
        return img

