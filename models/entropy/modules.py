import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd

from ..utils import conv3x3, conv5x5, subpel_conv3x3
from ..arch_utils import ResBlock

class HyperEncoder(nn.Module):
    def __init__(self, c_in=192, c_out=192, abs_flag=True):
        super(HyperEncoder, self).__init__()
        self.conv1 = conv3x3(c_in, c_out)
        self.conv2 = conv5x5(c_out, c_out)
        self.conv3 = conv5x5(c_out, c_out)

        self.abs_flag = abs_flag

    def forward(self, x):
        if self.abs_flag:
            x = x.abs()
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.conv3(x)
        return x


class HyperDecoder(nn.Module):
    def __init__(self, c_in=192, c_out=384, ks=5):
        super(HyperDecoder, self).__init__()
        _conv = partial(nn.ConvTranspose2d,
                        kernel_size=ks,
                        stride=2,
                        padding=(ks - 1) // 2,
                        output_padding=ks % 2,
                        bias=False)
        c_add = c_out - c_in
        self.conv1 = _conv(c_in, c_in)
        self.conv2 = _conv(c_in, c_in+c_add//2)
        self.conv3 = conv3x3(c_in+c_add//2, c_out)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.conv3(x)
        return x


class HyperEncoderLight(nn.Module):
    def __init__(self, N=192):
        super(HyperEncoderLight, self).__init__()
        self.encoder = nn.Sequential(
            conv3x3(N, N, bias=True),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, bias=True),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2, bias=True),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, bias=True),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2, bias=True),
        )


    def forward(self, x):
        x = self.encoder(x)
        return x


class HyperDecoderLight(nn.Module):
    def __init__(self, N=192):
        super(HyperDecoderLight, self).__init__()
        self.decoder = nn.Sequential(
            conv3x3(N, N, bias=True),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2, bias=True),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2, bias=True),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


class HyperEncoderLightConv(nn.Module):
    def __init__(self, N=192, c_out=-1):
        super(HyperEncoderLightConv, self).__init__()
        if c_out==-1:
            c_out = N
        self.encoder = nn.Sequential(
            nn.Conv2d(N, c_out, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(c_out, c_out, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(c_out, c_out, 3, stride=2, padding=1),
        )


    def forward(self, x):
        x = self.encoder(x)
        return x


class HyperDecoderLightConv(nn.Module):
    def __init__(self, N=192):
        super(HyperDecoderLightConv, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(N, N, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(N, N, 3,stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(N, N, 3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


class GatherLight(nn.Module):
    def __init__(self, c_in=128, c_out=256):
        super(GatherLight, self).__init__()
        self.gather = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride=1, padding=1),
            ResBlock(c_out, bottleneck=True)
        )

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, x):
        x = self.gather(x)
        return x