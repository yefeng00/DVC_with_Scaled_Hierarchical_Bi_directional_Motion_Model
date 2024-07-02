import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .modules import *
from .entropy_coder import get_entropy_coder
from .probability import GaussianProbModel, FactorizedProbModel, FactorizedProbModel2, LaplacianProbModel
from ..activation import GDN


def conv3x3(in_ch, out_ch, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)

def subpel_conv1x1(in_ch, out_ch, r=1):
    """1x1 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=1, padding=0), nn.PixelShuffle(r)
    )


class EntropyModel(nn.Module):
    def __init__(self):
        super(EntropyModel, self).__init__()
        self.coder = None
        self.coder_offset = 512
        self.is_train = False #not is_eval()
        self.ed_time = 0
        if not self.is_train:
            try:
                self.coder = get_entropy_coder()
            except:
                pass

    def flush(self):
        return self.coder.flush_encoder()

    def reset(self):
        return self.coder.reset_encoder()

    def set_stream(self, stream):
        return self.coder.set_stream(stream)

    def forward(self, *args, mode='forward', **kwargs):
        assert mode in {'forward', 'encode', 'decode'}
        if mode == 'encode':
            return self.encode(*args, **kwargs)
        elif mode == 'decode':
            return self.decode(*args, **kwargs)
        return self.forward_core(*args, **kwargs)

    def quantize(self, x, mean=None, noise=False):
        if mean is not None:
            x = x - mean
        if not self.is_train:
            x = x.round()
            return x + mean if mean is not None else x
        if noise:
            noise = torch.nn.init.uniform_(torch.zeros_like(x), -0.4, 0.4)
            noise = noise.clone().detach()
            x = x + noise
            # x = x + torch.empty_like(x).uniform_(-0.4, 0.4)
        else:
            n = torch.round(x) - x
            n = n.clone().detach()
            x = x + n
        return x + mean if mean is not None else x


class HyperEntropyLight(EntropyModel):
    def __init__(self, c_in=192):
        super(HyperEntropyLight, self).__init__()
        self.hyper_encoder = HyperEncoderLight(c_in)
        self.hyper_decoder = HyperDecoderLight(c_in)
        self.hyper_entropy = FactorizedProbModel(c_in)
        self.entropy = GaussianProbModel()

    def forward_core(self, y):
        # hyper
        z = self.hyper_encoder(y)
        # quantization z
        z_hat, z_likelihoods = self.hyper_entropy(z)
        scales_hat, means_hat = self.hyper_decoder(z_hat).chunk(2, 1)
        # compute likelihoods
        y_hat, y_likelihoods = self.entropy(y, scales_hat, means_hat)
        # y_hat, y_likelihoods = self.entropy(y, scales_hat, means_hat)
        return y_likelihoods, z_likelihoods, y_hat

    def encode(self, y):
        # hyper
        z = self.hyper_encoder(y)
        # quantization z
        z_hat, z_likelihoods = self.hyper_entropy(z)
        self.hyper_entropy.update()
        self.hyper_entropy.compress(z, self.coder)
        scales_hat, means_hat = self.hyper_decoder(z_hat).chunk(2, 1)
        # compute likelihoods
        y_hat, y_likelihoods = self.entropy(y, scales_hat, means=means_hat)
        self.entropy.update()
        indexes = self.entropy.build_indexes(scales_hat)
        self.entropy.compress(y, indexes, means_hat, self.coder)
        return y_likelihoods, z_likelihoods, y_hat

    def decode(self, h, w):
        yh, yw = h // 16, w // 16
        zh, zw = h // 64, w // 64
        self.hyper_entropy.update()
        z_hat = self.hyper_entropy.decompress((zh, zw), self.coder)
        scales_hat, means_hat = self.hyper_decoder(z_hat).chunk(2, 1)
        self.entropy.update()
        indexes = self.entropy.build_indexes(scales_hat)
        y_hat = self.entropy.decompress(indexes, means_hat, self.coder)
        return y_hat

class ConditionHyperPriorEntropy(EntropyModel):
    def __init__(self, c_in=128, c_prior=128, skip=0.0, quant=-1, warmup=[0, 0]):
        super(ConditionHyperPriorEntropy, self).__init__()
        self.hyper_encoder = HyperEncoderLightConv(c_in+c_prior, c_in)
        self.hyper_decoder = HyperDecoderLightConv(c_in)
        self.hyper_entropy = FactorizedProbModel2(c_in)
        self.gather = GatherLight(c_in=c_in+c_prior,c_out=c_in*2)
        self.entropy = GaussianProbModel()
        self.skip = skip
        self.warmup = warmup
        self.quant = quant
        if self.quant > 0:
            self.gather = GatherLight(c_in=c_in+c_prior,c_out=c_in*3)
        
    def set_skip_ratio(self, skip):
        self.skip = skip

    def skip_process(self, y_hat, y_likelihoods, mean, scale, w_idx=0):
        if self.skip is not None:
            skip_rate = self.skip
            if self.is_train and self.warmup[w_idx] > 0:
                skip_rate = 0
                self.warmup[w_idx] -= 1
            skip_mask = (scale.abs() < skip_rate).float()
            #print(torch.mean(skip_mask))
            y_hat = y_hat * (1-skip_mask) + mean * skip_mask
            y_likelihoods = y_likelihoods * (1-skip_mask) + skip_mask
        return y_hat, y_likelihoods

    def skip_encode(self, y_hat, mean, scale, y_likelihoods):
        if self.skip is not None:
            skip_rate = self.skip
            skip_mask = (scale.abs() < skip_rate).float()
            y_hat = y_hat * (1-skip_mask) + mean * skip_mask
            y_compress = y_hat[skip_mask==0]
            mean_compress = mean[skip_mask==0]
            scale_compress = scale[skip_mask==0]
            y_likelihoods = y_likelihoods * (1-skip_mask) + skip_mask
        return y_hat, y_compress, mean_compress, scale_compress, y_likelihoods

    def skip_decode(self, mean, scale):
        if self.skip is not None:
            skip_rate = self.skip
            skip_mask = (scale.abs() < skip_rate).float()
            mean_compress = mean[skip_mask==0]
            scale_compress = scale[skip_mask==0]
        return mean_compress, scale_compress, skip_mask

    def forward_core(self, y, *prior):
        if prior[-1] is None:
            prior, w_idx = (*prior[:-1], torch.zeros_like(y).to(y.device)), 0
        else:
            w_idx = 1
        # hyper prior
        z = self.hyper_encoder(torch.cat([y, *prior], dim=1))
        z_hat, z_tilde = self.quantize(z), self.quantize(z, noise=True)
        psi = self.hyper_decoder(z_hat)
        # entropy model estiamtor
        dist_data = self.gather(torch.cat([psi, *prior], dim=1))
        if self.quant > 0:
            mean, scale, quant = dist_data.chunk(3, dim=1)
            quant = quant.abs().clamp(min=self.quant)
        else:
            mean, scale = dist_data.chunk(2, dim=1)
            quant = torch.ones_like(mean)
        y, mean, scale = y / quant, mean / quant, scale / quant
        # compute likelihoods
        z_likelihoods = self.hyper_entropy(z_tilde)
        y_hat, y_likelihoods = self.entropy(y, scale.abs(), mean)
        # skip
        y_hat, y_likelihoods = self.skip_process(y_hat, y_likelihoods, mean, scale, w_idx)
        return y_likelihoods, z_likelihoods, y_hat * quant

    def encode(self, y, *prior):
        if self.ed_time == 0:
            self.entropy.update()
            self.hyper_entropy.update(True)
            self.ed_time = 1
        if prior[-1] is None:
            prior, w_idx = (*prior[:-1], torch.zeros_like(y).to(y.device)), 0
        # hyper
        z = self.hyper_encoder(torch.cat([y, *prior], dim=1))
        # quantization z
        z_hat = z.round()
        z_likelihoods = self.hyper_entropy(z_hat)
        self.hyper_entropy.compress(z_hat, self.coder)
        psi = self.hyper_decoder(z_hat)
        # entropy model estiamtor
        dist_data = self.gather(torch.cat([psi, *prior], dim=1))
        if self.quant > 0:
            mean, scale, quant = dist_data.chunk(3, dim=1)
            quant = quant.abs().clamp(min=self.quant)
        else:
            mean, scale = dist_data.chunk(2, dim=1)
            quant = torch.ones_like(mean).to(mean.device)
        # quantitize
        y, mean, scale = y / quant, mean / quant, scale / quant
        # compute likelihoods
        y_hat, y_likelihoods = self.entropy(y, scale.abs(), means=mean)
        # skip
        y_hat, y_compress, mean_compress, scale_compress, y_likelihoods = self.skip_encode(y_hat, mean, scale, y_likelihoods)
        indexes = self.entropy.build_indexes(scale_compress.abs())
        #print(indexes.shape)
        self.entropy.compress(y_compress, indexes, mean_compress, self.coder)
        return y_likelihoods, z_likelihoods, y_hat * quant

    def decode(self, h, w, *prior):
        if self.ed_time == 0:
            self.entropy.update()
            self.hyper_entropy.update(True)
            self.ed_time = 1
        # hyper
        zh, zw = h // 64, w // 64
        self.hyper_entropy.update(True)
        z_hat = self.hyper_entropy.decompress((zh, zw), self.coder)
        z_hat = z_hat.to(next(self.hyper_encoder.parameters()).device)
        psi = self.hyper_decoder(z_hat)
        if prior[-1] is None:
            prior = (*prior[:-1], torch.zeros_like(psi).to(psi.device))
        dist_data = self.gather(torch.cat([psi, *prior], dim=1))
        if self.quant > 0:
            mean, scale, quant = dist_data.chunk(3, dim=1)
            quant = quant.abs().clamp(min=self.quant)
        else:
            mean, scale = dist_data.chunk(2, dim=1)
            quant = torch.ones_like(mean).to(mean.device)
        # quantitize
        mean, scale = mean / quant, scale / quant
        #skip
        mean_compress, scale_compress, skip_mask = self.skip_decode(mean, scale)
        indexes = self.entropy.build_indexes(scale_compress.abs())
        y_compress = self.entropy.decompress(indexes, mean_compress, self.coder)
        y_hat = mean

        y_hat[skip_mask==0] = y_compress
        # unquantize
        y_hat = y_hat * quant
        return y_hat
    


class MvConditionHyperPriorEntropy(nn.Module):
    def __init__(self, c_in=128, c_prior=128, skip=0.0, quant=-1, warmup=0):
        super(MvConditionHyperPriorEntropy, self).__init__()
        c_mid = c_in + (c_prior) // 2
        self.warmup = warmup
        self.mv_hyper_prior_encoder = nn.Sequential(
            conv3x3(c_in, c_in),
            nn.LeakyReLU(),
            conv3x3(c_in, c_in),
            nn.LeakyReLU(),
            conv3x3(c_in, c_in, stride=2),
            nn.LeakyReLU(),
            conv3x3(c_in, c_in),
            nn.LeakyReLU(),
            conv3x3(c_in, c_in, stride=2),
        )

        self.mv_hyper_prior_decoder = nn.Sequential(
            conv3x3(c_in, c_in),
            nn.LeakyReLU(),
            subpel_conv1x1(c_in, c_in, 2),
            nn.LeakyReLU(),
            conv3x3(c_in, c_in * 3 // 2),
            nn.LeakyReLU(),
            subpel_conv1x1(c_in * 3 // 2, c_in * 3 // 2, 2),
            nn.LeakyReLU(),
            conv3x3(c_in * 3 // 2, c_in * 2),
        )
        self.mv_y_prior_fusion = nn.Sequential(
            nn.Conv2d(c_in * 4, c_in * 4, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(c_in * 4, c_in * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(c_in * 3, c_in * 3, 3, stride=1, padding=1)
        )
        self.hyper_entropy = FactorizedProbModel2(c_in)
        self.entropy = GaussianProbModel()
        self.is_train = False#not is_eval()
            
    def quantize(self, x, mean=None, noise=False):
        if mean is not None:
            x = x - mean
        if not self.is_train:
            x = x.round()
            return x + mean if mean is not None else x
        if noise:
            noise = torch.nn.init.uniform_(torch.zeros_like(x), -0.4, 0.4)
            noise = noise.clone().detach()
            x = x + noise
            # x = x + torch.empty_like(x).uniform_(-0.4, 0.4)
        else:
            n = torch.round(x) - x
            n = n.clone().detach()
            x = x + n
        return x + mean if mean is not None else x
    
    def skip_process(self, y_hat, y_likelihoods, mean, scale):
        skip_rate = 0.2
        if self.is_train and self.warmup > 0:
            skip_rate = 0
            self.warmup -= 1
        skip_mask = (scale.abs() < skip_rate).float()
        #print(torch.mean(skip_mask))
        y_hat = y_hat * (1-skip_mask) + mean * skip_mask
        y_likelihoods = y_likelihoods * (1-skip_mask) + skip_mask
        return y_hat, y_likelihoods
    
    def forward(self, y, prior_l, prior_r):
        if prior_l is None:
            if prior_r is None:
                prior_l = torch.zeros_like(y).to(y.device)
                prior_r = torch.zeros_like(y).to(y.device)
            else:
                prior_l = -prior_r
        elif prior_r is None:
            prior_r = -prior_l
        mv_z = self.mv_hyper_prior_encoder(y)
        mv_z_hat, mv_z_hat1 = self.quantize(mv_z), self.quantize(mv_z, noise=True)
        mv_params = self.mv_hyper_prior_decoder(mv_z_hat)
        
        mv_params = torch.cat((mv_params, prior_l, prior_r), dim=1)
        mv_scales_hat, mv_means_hat, quant = self.mv_y_prior_fusion(mv_params).chunk(3, 1)
        quant = quant.abs().clamp(min=1)
        y, mv_means_hat, mv_scales_hat = y / quant, mv_means_hat / quant, mv_scales_hat / quant
        
        z_likelihoods = self.hyper_entropy(mv_z_hat1)
        y_hat, y_likelihoods = self.entropy(y, mv_scales_hat.abs(), mv_means_hat)
        # skip
        y_hat, y_likelihoods = self.skip_process(y_hat, y_likelihoods, mv_means_hat, mv_scales_hat)
        return y_likelihoods, z_likelihoods, y_hat * quant

class ResConditionHyperPriorEntropy(nn.Module):
    def __init__(self, c_in=128, c_prior=128, warmup=0):
        super(ResConditionHyperPriorEntropy, self).__init__()
        c_mid = c_in + (c_prior) // 2
        self.warmup = warmup
        self.contextual_hyper_prior_encoder = nn.Sequential(
            nn.Conv2d(c_in, c_in, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(c_in, c_in, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(c_in, c_in, 3, stride=2, padding=1),
        )
        self.contextual_hyper_prior_decoder = nn.Sequential(
            nn.ConvTranspose2d(c_in, c_in, 3,
                               stride=2, padding=1, output_padding=1),
            #nn.ConvTranspose2d(c_in, c_in * 3 // 2, 3,
            #                   stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(c_in, c_in * 3 // 2, 3,
                               stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(c_in * 3 // 2, c_in * 2, 3, stride=1, padding=1)
        )
        self.entropy = GaussianProbModel()
        self.hyper_entropy = FactorizedProbModel2(c_in)
        self.temporal_prior_encoder = ContextualDecoder(c_in // 2, c_in)
        self.contextual_entropy_parameter = nn.Sequential(
            nn.Conv2d(c_in * 4, c_in * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(c_in * 3, c_in * 2, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(c_in * 2, c_in * 2, 3, stride=1, padding=1),
        )
        self.is_train = False#not is_eval()
         
    def quantize(self, x, mean=None, noise=False):
        if mean is not None:
            x = x - mean
        if not self.is_train:
            x = x.round()
            return x + mean if mean is not None else x
        if noise:
            noise = torch.nn.init.uniform_(torch.zeros_like(x), -0.4, 0.4)
            noise = noise.clone().detach()
            x = x + noise
            # x = x + torch.empty_like(x).uniform_(-0.4, 0.4)
        else:
            n = torch.round(x) - x
            n = n.clone().detach()
            x = x + n
        return x + mean if mean is not None else x
    
    def skip_process(self, y_hat, y_likelihoods, mean, scale):
        skip_rate = 0.2
        if self.is_train and self.warmup > 0:
            skip_rate = 0
            self.warmup -= 1
        skip_mask = (scale.abs() < skip_rate).float()
        #print(torch.mean(skip_mask))
        y_hat = y_hat * (1-skip_mask) + mean * skip_mask
        y_likelihoods = y_likelihoods * (1-skip_mask) + skip_mask
        return y_hat, y_likelihoods
    
    def forward(self, y, res_tprior, context1, context2, context3):
        if res_tprior is None:
            res_tprior = torch.zeros_like(y).to(y.device)
            
        z = self.contextual_hyper_prior_encoder(y)
        z_hat, z_hat1 = self.quantize(z), self.quantize(z, noise=True)
        hierarchical_params = self.contextual_hyper_prior_decoder(z_hat)
        temporal_params = self.temporal_prior_encoder(context1, context2, context3)

        params = torch.cat((temporal_params, hierarchical_params, res_tprior), dim=1)
        gaussian_params = self.contextual_entropy_parameter(params)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        
        z_likelihoods = self.hyper_entropy(z_hat1)

        y_hat, y_likelihoods = self.entropy(y, scales_hat.abs(), means_hat)
        # skip
        y_hat, y_likelihoods = self.skip_process(y_hat, y_likelihoods, means_hat, scales_hat)
        return y_likelihoods, z_likelihoods, y_hat
    











class ContextualDecoder(nn.Module): #ok
    def __init__(self, c_in=64, c_out=128):
        super().__init__()
        c_mid = (c_in + c_out) // 2
        self.conv1 = nn.Conv2d(c_in, c_in, 3, stride=2, padding=1)
        self.gdn1 = GDN(c_in)
        self.conv2 = nn.Conv2d(c_in * 2, c_mid, 3, stride=2, padding=1)
        self.gdn2 = GDN(c_mid)
        self.conv3 = nn.Conv2d(c_mid + c_in, c_out, 3, stride=2, padding=1)
        self.gdn3 = GDN(c_out)
        self.conv4 = nn.Conv2d(c_out, c_out, 3, stride=2, padding=1)

    def forward(self, context1, context2, context3):
        feature = self.conv1(context1)
        feature = self.gdn1(feature)
        feature = self.conv2(torch.cat([feature, context2], dim=1))
        feature = self.gdn2(feature)
        feature = self.conv3(torch.cat([feature, context3], dim=1))
        feature = self.gdn3(feature)
        feature = self.conv4(feature)
        return feature
