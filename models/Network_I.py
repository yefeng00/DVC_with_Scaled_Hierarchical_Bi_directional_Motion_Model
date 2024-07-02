
import torch.nn as nn

import torch

from .layers2 import conv3x3, DepthConvBlock2, DepthConvBlock3, DepthConvBlock4, \
    ResidualBlockUpsample, ResidualBlockWithStride2
from .video_net2 import UNet
import math

from .entropy_models import BitEstimator, GaussianEncoder, EntropyCoder
from .stream_helper import get_padding_size

class IntraEncoder(nn.Module):
    def __init__(self, N=256, z_channel=128, inplace=False):
        super().__init__()

        self.enc_1 = nn.Sequential(
            ResidualBlockWithStride2(3, 128, inplace=inplace),
            DepthConvBlock3(128, 128, inplace=inplace),
        )
        self.enc_2 = nn.Sequential(
            ResidualBlockWithStride2(128, 192, inplace=inplace),
            DepthConvBlock3(192, 192, inplace=inplace),
            ResidualBlockWithStride2(192, N, inplace=inplace),
            DepthConvBlock3(N, N, inplace=inplace),
            nn.Conv2d(N, N, 3, stride=2, padding=1),
        )

    def forward(self, x, quant_step):
        out = self.enc_1(x)
        out = out * quant_step
        return self.enc_2(out)

class IntraDecoder(nn.Module):
    def __init__(self, N=256, z_channel=128, inplace=False):
        super().__init__()

        self.dec_1 = nn.Sequential(
            DepthConvBlock3(N, N, inplace=inplace),
            ResidualBlockUpsample(N, N, 2, inplace=inplace),
            DepthConvBlock3(N, N, inplace=inplace),
            ResidualBlockUpsample(N, 192, 2, inplace=inplace),
            DepthConvBlock3(192, 192, inplace=inplace),
            ResidualBlockUpsample(192, 128, 2, inplace=inplace),
        )
        self.dec_2 = nn.Sequential(
            DepthConvBlock3(128, 128, inplace=inplace),
            ResidualBlockUpsample(128, 16, 2, inplace=inplace),
        )

    def forward(self, x, quant_step):
        out = self.dec_1(x)
        out = out * quant_step
        return self.dec_2(out)


class HyperEncoder(nn.Module):
    def __init__(self, N=256, z_channel=128, inplace=False):
        super().__init__()
        self.hyper_enc = nn.Sequential(
            DepthConvBlock4(N, z_channel, inplace=inplace),
            nn.Conv2d(z_channel, z_channel, 3, stride=2, padding=1),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv2d(z_channel, z_channel, 3, stride=2, padding=1),
        )
    def forward(self, x):
        return self.hyper_enc(x)

class HyperDecoder(nn.Module):
    def __init__(self, N=256, z_channel=128, inplace=False):
        super().__init__()
        self.hyper_dec = nn.Sequential(
            ResidualBlockUpsample(z_channel, z_channel, 2, inplace=inplace),
            ResidualBlockUpsample(z_channel, z_channel, 2, inplace=inplace),
            DepthConvBlock4(z_channel, N),
        )
    def forward(self, x):
        return self.hyper_dec(x)

class PriorFusionY(nn.Module):
    def __init__(self, N=256, z_channel=128, inplace=False):
        super().__init__()
        self.y_prior_fusion = nn.Sequential(
            DepthConvBlock4(N, N * 2, inplace=inplace),
            DepthConvBlock4(N * 2, N * 2 + 2, inplace=inplace),
        )
    def forward(self, x):
        return self.y_prior_fusion(x)


class SpatialPriorReduction(nn.Module):
    def __init__(self, N=256, z_channel=128, inplace=False):
        super().__init__()
        self.y_spatial_prior_reduction = nn.Conv2d(N * 2 + 2, N * 1, 1)
    def forward(self, x):
        return self.y_spatial_prior_reduction(x)

class SpatialPriorAdaptor1(nn.Module):
    def __init__(self, N=256, z_channel=128, inplace=False):
        super().__init__()
        self.y_spatial_prior_adaptor_1 = DepthConvBlock2(N * 2, N * 2, inplace=inplace)
    def forward(self, x):
        return self.y_spatial_prior_adaptor_1(x)

class SpatialPriorAdaptor2(nn.Module):
    def __init__(self, N=256, z_channel=128, inplace=False):
        super().__init__()
        self.y_spatial_prior_adaptor_2 = DepthConvBlock2(N * 2, N * 2, inplace=inplace)
    def forward(self, x):
        return self.y_spatial_prior_adaptor_2(x)


class SpatialPriorAdaptor3(nn.Module):
    def __init__(self, N=256, z_channel=128, inplace=False):
        super().__init__()
        self.y_spatial_prior_adaptor_3 = DepthConvBlock2(N * 2, N * 2, inplace=inplace)
    def forward(self, x):
        return self.y_spatial_prior_adaptor_3(x)

class SpatialPrior(nn.Module):
    def __init__(self, N=256, z_channel=128, inplace=False):
        super().__init__()
        self.y_spatial_prior = nn.Sequential(
            DepthConvBlock2(N * 2, N * 2, inplace=inplace),
            DepthConvBlock2(N * 2, N * 2, inplace=inplace),
            DepthConvBlock2(N * 2, N * 2, inplace=inplace),
        )
    def forward(self, x):
        return self.y_spatial_prior(x)

class Refine(nn.Module):
    def __init__(self, N=256, z_channel=128, inplace=False):
        super().__init__()
        self.refine = nn.Sequential(
            UNet(16, 16, inplace=inplace),
            conv3x3(16, 3),
        )
    def forward(self, x):
        return self.refine(x)

class ScaleEnc(nn.Module):
    def __init__(self, N=256, z_channel=128, inplace=False):
        super().__init__()
        self.q_scale_enc = nn.Parameter(torch.ones((self.get_qp_num(), 128, 1, 1)))
    
    @staticmethod
    def get_qp_num():
        return 64
    def forward(self, x):
        return self.q_scale_enc(x)

class ScaleDec(nn.Module):
    def __init__(self, N=256, z_channel=128, inplace=False):
        super().__init__()
        self.q_scale_dec = nn.Parameter(torch.ones((self.get_qp_num(), 128, 1, 1)))
    
    @staticmethod
    def get_qp_num():
        return 64
    def forward(self, x):
        return self.q_scale_dec(x)

    
class IFrame(nn.Module):
    def __init__(self):
        super().__init__()
        self.ec_thread = False
        self.stream_part = 1

        self.masks = {}
        self.enc = IntraEncoder()
        self.hyper_enc = HyperEncoder()
        self.hyper_dec = HyperDecoder()
        self.y_prior_fusion = PriorFusionY()
        self.y_spatial_prior_reduction = SpatialPriorReduction()
        self.y_spatial_prior_adaptor_1 = SpatialPriorAdaptor1()
        self.y_spatial_prior_adaptor_2 = SpatialPriorAdaptor2()
        self.y_spatial_prior_adaptor_3 = SpatialPriorAdaptor3()
        self.y_spatial_prior = SpatialPrior()
        self.dec = IntraDecoder()
        self.refine = Refine()
        self.q_scale_enc = ScaleEnc()
        self.q_scale_dec = ScaleDec()
        self.bit_estimator_z = BitEstimator()

    def forward(self, x, q_index=63):
        _, _, H, W = x.size()
        device = x.device
        index = self.get_index_tensor(q_index, device)
        curr_q_enc = torch.index_select(self.q_scale_enc.q_scale_enc, 0, index)
        curr_q_dec = torch.index_select(self.q_scale_dec.q_scale_dec, 0, index)

        y = self.enc(x, curr_q_enc)
        #print(y)
        y_pad, slice_shape = self.pad_for_y(y)
        z = self.hyper_enc(y_pad)
        z_q = self.quant(z)
        z_hat = z_q
        #print(z_q)
        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        params = self.slice_to_y(params, slice_shape)
        y_res, y_q, y_hat, scales_hat = self.forward_four_part_prior(
            y, params,
            self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
            self.y_spatial_prior_adaptor_3, self.y_spatial_prior,
            y_spatial_prior_reduction=self.y_spatial_prior_reduction)

        x_hat = self.dec(y_hat, curr_q_dec)
        x_hat = self.refine(x_hat)

        y_for_bit = y_q
        z_for_bit = z_q
        bits_y = self.get_y_gaussian_bits(y_for_bit, scales_hat)
        bits_z = self.get_z_bits(z_for_bit, self.bit_estimator_z, index)
        pixel_num = H * W
        bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num
        bpp_z = torch.sum(bits_z, dim=(1, 2, 3)) / pixel_num

        bits = torch.sum(bpp_y + bpp_z) * pixel_num
        #bits = torch.sum(bpp_y + bpp_z)

        return {
            "x_hat": x_hat,
            "bit": bits,
        }
    
    def quant(self, x):
        return torch.round(x)

    def get_one_q_scale(self, q_scale, q_index):
        min_q = q_scale[0:1, :, :, :]
        max_q = q_scale[1:2, :, :, :]
        step = (torch.log(max_q) - torch.log(min_q)) / (self.get_qp_num() - 1)
        q = torch.exp(torch.log(min_q) + step * q_index)
        return q

    def get_curr_q(self, q_scale, q_index):
        if isinstance(q_index, list):
            q_step = [self.get_one_q_scale(q_scale, i) for i in q_index]
            q_step = torch.cat(q_step, dim=0)
        else:
            q_step = self.get_one_q_scale(q_scale, q_index)

        return q_step

    @staticmethod
    def get_index_tensor(q_index, device):
        if not isinstance(q_index, list):
            q_index = [q_index]
        return torch.tensor(q_index, dtype=torch.int32, device=device)

    @staticmethod
    def get_qp_num():
        return 64


    @staticmethod
    def probs_to_bits(probs):
        factor = -1.0 / math.log(2.0)
        bits = torch.log(probs + 1e-5) * factor
        bits = torch.clamp(bits, 0, None)
        return bits

    def get_y_gaussian_bits(self, y, sigma):
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.normal.Normal(mu, sigma)
        probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
        probs = probs.to(torch.float32)
        return IFrame.probs_to_bits(probs)

    def get_y_laplace_bits(self, y, sigma):
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
        probs = probs.to(torch.float32)
        return IFrame.probs_to_bits(probs)

    def get_z_bits(self, z, bit_estimator, index):
        probs = bit_estimator.get_cdf(z + 0.5, index) - bit_estimator.get_cdf(z - 0.5, index)
        probs = probs.to(torch.float32)
        return IFrame.probs_to_bits(probs)

    def update(self, force=False):
        self.entropy_coder = EntropyCoder(self.ec_thread, self.stream_part)
        self.gaussian_encoder.update(force=force, entropy_coder=self.entropy_coder)
        self.bit_estimator_z.update(force=force, entropy_coder=self.entropy_coder)
        if self.bit_estimator_z_mv is not None:
            self.bit_estimator_z_mv.update(force=force, entropy_coder=self.entropy_coder)

    def pad_for_y(self, y):
        _, _, H, W = y.size()
        padding_l, padding_r, padding_t, padding_b = get_padding_size(H, W, 4)
        y_pad = torch.nn.functional.pad(
            y,
            (padding_l, padding_r, padding_t, padding_b),
            mode="replicate",
        )
        return y_pad, (-padding_l, -padding_r, -padding_t, -padding_b)

    @staticmethod
    def get_to_y_slice_shape(height, width):
        padding_l, padding_r, padding_t, padding_b = get_padding_size(height, width, 4)
        return (-padding_l, -padding_r, -padding_t, -padding_b)

    def slice_to_y(self, param, slice_shape):
        return torch.nn.functional.pad(param, slice_shape)

    @staticmethod
    def separate_prior(params, is_video=False):
        if is_video:
            quant_step, scales, means = params.chunk(3, 1)
            quant_step = torch.clamp(quant_step, 0.5, None)
            q_enc = 1. / quant_step
            q_dec = quant_step
        else:
            q = params[:, :2, :, :]
            q_enc, q_dec = (torch.sigmoid(q) * 1.5 + 0.5).chunk(2, 1)
            scales, means = params[:, 2:, :, :].chunk(2, 1)
        return q_enc, q_dec, scales, means

    def get_mask(self, height, width, dtype, device):
        curr_mask_str = f"{width}x{height}"
        if curr_mask_str not in self.masks:
            micro_mask = torch.tensor(((1, 0), (0, 1)), dtype=dtype, device=device)
            mask_0 = micro_mask.repeat((height + 1) // 2, (width + 1) // 2)
            mask_0 = mask_0[:height, :width]
            mask_0 = torch.unsqueeze(mask_0, 0)
            mask_0 = torch.unsqueeze(mask_0, 0)
            mask_1 = torch.ones_like(mask_0) - mask_0
            self.masks[curr_mask_str] = [mask_0, mask_1]
        return self.masks[curr_mask_str]

    def process_with_mask(self, y, scales, means, mask):
        scales_hat = scales * mask
        means_hat = means * mask

        y_res = (y - means_hat) * mask
        y_q = self.quant(y_res)
        y_hat = y_q + means_hat

        return y_res, y_q, y_hat, scales_hat

    @staticmethod
    def get_one_channel_four_parts_mask(height, width, dtype, device):
        micro_mask_0 = torch.tensor(((1, 0), (0, 0)), dtype=dtype, device=device)
        mask_0 = micro_mask_0.repeat((height + 1) // 2, (width + 1) // 2)
        mask_0 = mask_0[:height, :width]
        mask_0 = torch.unsqueeze(mask_0, 0)
        mask_0 = torch.unsqueeze(mask_0, 0)

        micro_mask_1 = torch.tensor(((0, 1), (0, 0)), dtype=dtype, device=device)
        mask_1 = micro_mask_1.repeat((height + 1) // 2, (width + 1) // 2)
        mask_1 = mask_1[:height, :width]
        mask_1 = torch.unsqueeze(mask_1, 0)
        mask_1 = torch.unsqueeze(mask_1, 0)

        micro_mask_2 = torch.tensor(((0, 0), (1, 0)), dtype=dtype, device=device)
        mask_2 = micro_mask_2.repeat((height + 1) // 2, (width + 1) // 2)
        mask_2 = mask_2[:height, :width]
        mask_2 = torch.unsqueeze(mask_2, 0)
        mask_2 = torch.unsqueeze(mask_2, 0)

        micro_mask_3 = torch.tensor(((0, 0), (0, 1)), dtype=dtype, device=device)
        mask_3 = micro_mask_3.repeat((height + 1) // 2, (width + 1) // 2)
        mask_3 = mask_3[:height, :width]
        mask_3 = torch.unsqueeze(mask_3, 0)
        mask_3 = torch.unsqueeze(mask_3, 0)

        return mask_0, mask_1, mask_2, mask_3

    def get_mask_four_parts(self, batch, channel, height, width, dtype, device):
        curr_mask_str = f"{batch}_{channel}x{width}x{height}"
        with torch.no_grad():
            if curr_mask_str not in self.masks:
                assert channel % 4 == 0
                m = torch.ones((batch, channel // 4, height, width), dtype=dtype, device=device)
                m0, m1, m2, m3 = self.get_one_channel_four_parts_mask(height, width, dtype, device)

                mask_0 = torch.cat((m * m0, m * m1, m * m2, m * m3), dim=1)
                mask_1 = torch.cat((m * m3, m * m2, m * m1, m * m0), dim=1)
                mask_2 = torch.cat((m * m2, m * m3, m * m0, m * m1), dim=1)
                mask_3 = torch.cat((m * m1, m * m0, m * m3, m * m2), dim=1)

                self.masks[curr_mask_str] = [mask_0, mask_1, mask_2, mask_3]
        return self.masks[curr_mask_str]

    @staticmethod
    def combine_four_parts(x_0_0, x_0_1, x_0_2, x_0_3,
                           x_1_0, x_1_1, x_1_2, x_1_3,
                           x_2_0, x_2_1, x_2_2, x_2_3,
                           x_3_0, x_3_1, x_3_2, x_3_3):
        x_0 = x_0_0 + x_0_1 + x_0_2 + x_0_3
        x_1 = x_1_0 + x_1_1 + x_1_2 + x_1_3
        x_2 = x_2_0 + x_2_1 + x_2_2 + x_2_3
        x_3 = x_3_0 + x_3_1 + x_3_2 + x_3_3
        return torch.cat((x_0, x_1, x_2, x_3), dim=1)

    @staticmethod
    def combine_for_writing(x):
        x0, x1, x2, x3 = x.chunk(4, 1)
        return (x0 + x1) + (x2 + x3)

    def forward_four_part_prior(self, y, common_params,
                                y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                y_spatial_prior_adaptor_3, y_spatial_prior,
                                y_spatial_prior_reduction=None, write=False):
        '''
        y_0 means split in channel, the 0/4 quater
        y_1 means split in channel, the 1/4 quater
        y_2 means split in channel, the 2/4 quater
        y_3 means split in channel, the 3/4 quater
        y_?_0, means multiply with mask_0
        y_?_1, means multiply with mask_1
        y_?_2, means multiply with mask_2
        y_?_3, means multiply with mask_3
        '''
        q_enc, q_dec, scales, means = self.separate_prior(common_params,
                                                          y_spatial_prior_reduction is None)
        if y_spatial_prior_reduction is not None:
            common_params = y_spatial_prior_reduction(common_params)
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, dtype, device)

        y = y * q_enc

        y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales, means, mask_0)

        y_hat_so_far = y_hat_0
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_1)

        y_hat_so_far = y_hat_so_far + y_hat_1
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        y_res_2, y_q_2, y_hat_2, s_hat_2 = self.process_with_mask(y, scales, means, mask_2)

        y_hat_so_far = y_hat_so_far + y_hat_2
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        y_res_3, y_q_3, y_hat_3, s_hat_3 = self.process_with_mask(y, scales, means, mask_3)

        y_res = (y_res_0 + y_res_1) + (y_res_2 + y_res_3)
        y_q = (y_q_0 + y_q_1) + (y_q_2 + y_q_3)
        y_hat = y_hat_so_far + y_hat_3
        scales_hat = (s_hat_0 + s_hat_1) + (s_hat_2 + s_hat_3)

        y_hat = y_hat * q_dec

        if write:
            y_q_w_0 = self.combine_for_writing(y_q_0)
            y_q_w_1 = self.combine_for_writing(y_q_1)
            y_q_w_2 = self.combine_for_writing(y_q_2)
            y_q_w_3 = self.combine_for_writing(y_q_3)
            scales_w_0 = self.combine_for_writing(s_hat_0)
            scales_w_1 = self.combine_for_writing(s_hat_1)
            scales_w_2 = self.combine_for_writing(s_hat_2)
            scales_w_3 = self.combine_for_writing(s_hat_3)
            return y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, \
                scales_w_0, scales_w_1, scales_w_2, scales_w_3, y_hat
        return y_res, y_q, y_hat, scales_hat

    def compress_four_part_prior(self, y, common_params,
                                 y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                 y_spatial_prior_adaptor_3, y_spatial_prior,
                                 y_spatial_prior_reduction=None):
        return self.forward_four_part_prior(y, common_params,
                                            y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                            y_spatial_prior_adaptor_3, y_spatial_prior,
                                            y_spatial_prior_reduction, write=True)

    def decompress_four_part_prior(self, common_params,
                                   y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                   y_spatial_prior_adaptor_3, y_spatial_prior,
                                   y_spatial_prior_reduction=None):
        _, quant_step, scales, means = self.separate_prior(common_params,
                                                           y_spatial_prior_reduction is None)
        if y_spatial_prior_reduction is not None:
            common_params = y_spatial_prior_reduction(common_params)
        dtype = means.dtype
        device = means.device
        B, C, H, W = means.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, dtype, device)

        scales_r = self.combine_for_writing(scales * mask_0)
        y_q_r = self.gaussian_encoder.decode_stream(scales_r, dtype, device)
        y_hat_curr_step = (torch.cat((y_q_r, y_q_r, y_q_r, y_q_r), dim=1) + means) * mask_0
        y_hat_so_far = y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        scales_r = self.combine_for_writing(scales * mask_1)
        y_q_r = self.gaussian_encoder.decode_stream(scales_r, dtype, device)
        y_hat_curr_step = (torch.cat((y_q_r, y_q_r, y_q_r, y_q_r), dim=1) + means) * mask_1
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        scales_r = self.combine_for_writing(scales * mask_2)
        y_q_r = self.gaussian_encoder.decode_stream(scales_r, dtype, device)
        y_hat_curr_step = (torch.cat((y_q_r, y_q_r, y_q_r, y_q_r), dim=1) + means) * mask_2
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        scales_r = self.combine_for_writing(scales * mask_3)
        y_q_r = self.gaussian_encoder.decode_stream(scales_r, dtype, device)
        y_hat_curr_step = (torch.cat((y_q_r, y_q_r, y_q_r, y_q_r), dim=1) + means) * mask_3
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        y_hat = y_hat_so_far * quant_step

        return y_hat
