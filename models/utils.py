import torch
import torch.nn as nn
from .logger import is_eval
from .activation import GDN
import numpy as np

def get_train_order(num):
    seq = []
    seq_diverse = [0 for _ in range(num)]
    left = 0
    right = num - 1
    now = 0
    while(left < right):
        seq.append(left)
        seq_diverse[left] = now
        now += 1
        seq.append(right)
        seq_diverse[right] = now
        now += 1
        left += 1
        right -= 1
        if left == right:
            seq.append(left)
            seq_diverse[left] = now
    return seq, seq_diverse

def get_act(in_planes=192, name='lrelu', inverse=False):
    if name == 'lrelu':
        return act_lrelu(in_planes)
    elif name == 'lrelu_0.1':
        return act_lrelu1(in_planes)
    elif name == 'relu':
        return act_relu(in_planes)
    elif name == 'bn3d':
        return nn.BatchNorm3d(in_planes)
    elif name == 'gdn':
        return GDN(in_planes, inverse)
    elif name == 'none':
        return nn.Identity()
    else:
        return nn.Identity()


def get_code_seq(gop):
    if gop <= 1:
        print('Wrong Gop')
    if np.log2(gop) % 1 != 0:
        print('Wrong Gop')
    sum_layer = int(np.log2(gop))
    layers = [0]#[0] * (gop + 1)
    refs = [0]#[0] * (gop + 1)
    for i in range(1, gop):
        layer = sum_layer - 1
        j = 2
        while i % j == 0:
            j = j * 2
            layer -= 1
        refs.append(int(j/2))
        layers.append(layer)
    coded = [1] + [0] * (gop - 1) + [1]
    seq = [gop]
    for i in range(1, gop):
        get_seq(i, coded, seq, refs)
    return seq, refs, layers

def get_seq(idx, coded, seq, refs):
    if coded[idx] == 1:
        return
    get_seq(idx - refs[idx], coded, seq, refs)
    get_seq(idx + refs[idx], coded, seq, refs)
    coded[idx] = 1
    seq.append(idx)

def act_lrelu(in_planes):
    return nn.LeakyReLU(negative_slope=0.01, inplace=True)


def act_lrelu1(in_planes):
    return nn.LeakyReLU(negative_slope=0.1, inplace=True)


def act_relu(in_planes):
    return nn.ReLU(inplace=True)


def get_hat(x):
    return x.detach().round() - x.detach() + x if not is_eval() else x.round()


def get_tilde(x):
    return x + torch.empty_like(x).uniform_(-0.5, 0.5) if not is_eval() else x.round()


def get_noise(x):
    return x + torch.normal(0, torch.zeros_like(x)+2).to(x.device)


def normalize(x):
    return x / 255


def denormalize(x):
    return clip_image(x * 255)


def normalize_img(x):
    return x / 255


def denormalize_img(x):
    return clip_image(x * 255)


def quantization(x, bits=8):
    x = x * (2**(bits-8))
    x = x.round()
    return x * (2**(8-bits))

def normalize_flow(x):
    return x / 10


def denormalize_flow(x):
    return x * 10


def normalize_res(x):
    return x / 100


def denormalize_res(x):
    return x * 100


def clip_image(x):
    return torch.clamp(x, 0, 255)


def clip_tensor(x, c_val=512):
    return torch.clamp(x, -c_val, c_val-1)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     padding=0,
                     bias=bias)


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=bias)


def conv3x3_t(in_planes, out_planes, stride=2, bias=False):
    return nn.ConvTranspose2d(in_planes,
                              out_planes,
                              kernel_size=3,
                              stride=stride,
                              padding=1,
                              output_padding=1,
                              bias=bias)


def conv5x5(in_planes, out_planes, stride=2, bias=False):
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=5,
                     stride=stride,
                     padding=2,
                     bias=bias)


def conv5x5_t(in_planes, out_planes, stride=2, bias=False):
    return nn.ConvTranspose2d(in_planes,
                              out_planes,
                              kernel_size=5,
                              stride=stride,
                              padding=2,
                              output_padding=1,
                              bias=bias)


def subpel_conv3x3(in_ch, out_ch, r=1):
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=3,
                  padding=1), nn.PixelShuffle(r)
    )
