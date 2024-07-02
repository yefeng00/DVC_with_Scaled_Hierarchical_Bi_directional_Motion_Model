from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation import GDN
from .layers import subpel_conv1x1, conv3x3, \
    ResidualBlock, ResidualBlockWithStride, ResidualBlockUpsample
from .video_net import ME_Spynet, flow_warp, ResBlock, bilineardownsacling, LowerBound, UNet, \
    get_enc_dec_models, get_hyper_enc_dec_models
from .align import Align2D
from .entropy.entropy_model import MvConditionHyperPriorEntropy, ResConditionHyperPriorEntropy

class MotionChoose(nn.Module): 
    def __init__(self, c_in=64, c_out=3):
        super().__init__()
        self.conv4 = nn.Conv2d(3, 64, 3, stride=1, padding=1)

    def forward(self, pixel_mul):
        feature_mul = self.conv4(pixel_mul)
        return feature_mul

class MotionRec(nn.Module): 
    def __init__(self, c_in=4, N=48, c_out=2, ks=3):
        super(MotionRec, self).__init__()
        self.feature_conv = nn.Sequential(
            nn.Conv2d(c_in, N, 3, stride=1, padding=1),
            ResBlock(N),
            ResBlock(N),
        )
        self.recon_conv = nn.Conv2d(N, c_out, 3, stride=1, padding=1)
        #self.recon_conv = nn.ConvTranspose2d(N, c_out, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        feature = self.feature_conv(x)
        recon = self.recon_conv(feature)
        return recon

class FeatExtractor(nn.Module): #ok
    r"""The feature extraation module.
    """
    def __init__(self, c_in=3, c_out=64, stride=1, ks=3):
        super(FeatExtractor, self).__init__()
        self.layer = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1)

    def forward(self, x):
        x = self.layer(x)
        return x

class FeatureExtractor(nn.Module): #ok
    def __init__(self, channel=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        self.res_block1 = ResBlock(channel)
        self.conv2 = nn.Conv2d(channel, channel, 3, stride=2, padding=1)
        self.res_block2 = ResBlock(channel)
        self.conv3 = nn.Conv2d(channel, channel, 3, stride=2, padding=1)
        self.res_block3 = ResBlock(channel)

    def forward(self, feature):
        feature1 = self.conv1(feature)
        feature1 = self.res_block1(feature1)

        feature2 = self.conv2(feature1)
        feature2 = self.res_block2(feature2)

        feature3 = self.conv3(feature2)
        feature3 = self.res_block3(feature3)
        return feature1, feature2, feature3


class FeatRec(nn.Module): #ok
    r"""The feature extraation module.
    """
    def __init__(self, c_in=48, N=48, c_out=3, ks=3):
        super(FeatRec, self).__init__()
        self.feature_conv = nn.Sequential(
            nn.Conv2d(c_in, N, 3, stride=1, padding=1),
            ResBlock(N),
            ResBlock(N),
        )
        self.recon_conv = nn.Conv2d(N, c_out, 3, stride=1, padding=1)
        #self.recon_conv = nn.ConvTranspose2d(N, c_out, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        feature = self.feature_conv(x)
        recon = self.recon_conv(feature)
        return recon, feature

class EncoderResNet_3Layer(nn.Module): #ok
    r"""Encoder 3 down sample layer
    """

    def __init__(self, c_in=64, c_out=128, bottleneck=True, res_num=1):
        super(EncoderResNet_3Layer, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.conv1 = nn.Conv2d(c_in * 2, c_in, 3, stride=2, padding=1)
        self.gdn1 = GDN(c_in)
        self.res1 = ResBlock(c_in * 2, bottleneck=True, slope=0.1,
                             start_from_relu=False, end_with_relu=True)
        self.conv2 = nn.Conv2d(c_in * 2, c_in, 3, stride=2, padding=1)
        self.gdn2 = GDN(c_in)
        self.res2 = ResBlock(c_in * 2, bottleneck=True, slope=0.1,
                             start_from_relu=False, end_with_relu=True)
        self.conv3 = nn.Conv2d(c_in * 2, c_out, 3, stride=2, padding=1)
        self.gdn3 = GDN(c_out)
        self.conv4 = nn.Conv2d(c_out, c_out, 3, stride=2, padding=1)

    def forward(self, x, context1, context2, context3):
        feature = self.conv1(torch.cat([x, context1], dim=1))
        feature = self.gdn1(feature)
        feature = self.res1(torch.cat([feature, context2], dim=1))
        feature = self.conv2(feature)
        feature = self.gdn2(feature)
        feature = self.res2(torch.cat([feature, context3], dim=1))
        feature = self.conv3(feature)
        feature = self.gdn3(feature)
        feature = self.conv4(feature)
        return feature

class EncoderPrior_3Layer(nn.Module): #ok
    r"""Prior simple 3 downscample encoder
    """

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


class DecoderResNet_3Layer(nn.Module): #ok
    r"""Decoder with 3 upsample
    """

    def __init__(self, c_in=128, c_out=64, bottleneck=False, res_num=2):
        super(DecoderResNet_3Layer, self).__init__()
        self.up1 = nn.Sequential(nn.Conv2d(c_in, c_out * 2 ** 2, kernel_size=3, padding=1), nn.PixelShuffle(2))
        self.gdn1 = GDN(c_out, inverse=True)
        self.up2 = nn.Sequential(nn.Conv2d(c_out, c_out * 2 ** 2, kernel_size=3, padding=1), nn.PixelShuffle(2))
        self.gdn2 = GDN(c_out, inverse=True)
        self.res1 = ResBlock(c_out * 2, bottleneck=True, slope=0.1,
                             start_from_relu=False, end_with_relu=True)
        self.up3 = nn.Sequential(nn.Conv2d(c_out * 2, c_out * 2 ** 2, kernel_size=3, padding=1), nn.PixelShuffle(2))
        self.gdn3 = GDN(c_out, inverse=True)
        self.res2 = ResBlock(c_out * 2, bottleneck=True, slope=0.1,
                             start_from_relu=False, end_with_relu=True)
        self.up4 = nn.Sequential(nn.Conv2d(c_out * 2, c_out * 2 ** 2, kernel_size=3, padding=1), nn.PixelShuffle(2))

    def forward(self, feature, context2, context3):
        feature = self.up1(feature)
        feature = self.gdn1(feature)
        feature = self.up2(feature)
        feature = self.gdn2(feature)
        feature = self.res1(torch.cat([feature, context3], dim=1))
        feature = self.up3(feature)
        feature = self.gdn3(feature)
        feature = self.res2(torch.cat([feature, context2], dim=1))
        feature = self.up4(feature)
        return feature


class ContextRefine(nn.Module): #ok
    def __init__(self, c_in=48, N=48, bottleneck=False):
        super().__init__()

        self.conv1_1 = nn.Conv2d(c_in*2, c_in, 3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(c_in*2, c_in, 3, stride=1, padding=1)
        self.conv1_3 = nn.Conv2d(c_in*2, c_in, 3, stride=1, padding=1)
        
        self.conv3_up = nn.Sequential(nn.Conv2d(c_in, c_in * 2 ** 2, kernel_size=3, padding=1), nn.PixelShuffle(2))
        self.res_block3_up = ResBlock(c_in)
        self.conv3_out = nn.Conv2d(c_in, c_in, 3, padding=1)
        self.res_block3_out = ResBlock(c_in)
        self.conv2_up = nn.Sequential(nn.Conv2d(c_in * 2, c_in * 2 ** 2, kernel_size=3, padding=1), nn.PixelShuffle(2))
        self.res_block2_up = ResBlock(c_in)
        self.conv2_out = nn.Conv2d(c_in * 2, c_in, 3, padding=1)
        self.res_block2_out = ResBlock(c_in)
        self.conv1_out = nn.Conv2d(c_in * 2, c_in, 3, padding=1)
        self.res_block1_out = ResBlock(c_in)
        self.align = Align2D()

    def bilineardownsacling(self, inputfeature):
        inputheight = inputfeature.size()[2]
        inputwidth = inputfeature.size()[3]
        outfeature = F.interpolate(
            inputfeature, (inputheight // 2, inputwidth // 2), mode='bilinear', align_corners=False)
        return outfeature

    def forward(self, f_ref_rec_l, f_ref_rec_l_2, f_ref_rec_l_3, \
                      f_ref_rec_r, f_ref_rec_r_2, f_ref_rec_r_3, mv_rec_l, mv_rec_r, \
                      feature_mul_l, feature_mul_r):
        
        feature_mul_l_2 = self.bilineardownsacling(feature_mul_l) / 2
        feature_mul_l_3 = self.bilineardownsacling(feature_mul_l_2) / 2
        feature_mul_r_2 = self.bilineardownsacling(feature_mul_r) / 2
        feature_mul_r_3 = self.bilineardownsacling(feature_mul_r_2) / 2

        mv_rec_l_2 = self.bilineardownsacling(mv_rec_l) / 2
        mv_rec_l_3 = self.bilineardownsacling(mv_rec_l_2) / 2
        context_l_1, _ = self.align(f_ref_rec_l, mv_rec_l)
        context_l_2, _ = self.align(f_ref_rec_l_2, mv_rec_l_2)
        context_l_3, _ = self.align(f_ref_rec_l_3, mv_rec_l_3)

        mv_rec_r_2 = self.bilineardownsacling(mv_rec_r) / 2
        mv_rec_r_3 = self.bilineardownsacling(mv_rec_r_2) / 2
        context_r_1, _ = self.align(f_ref_rec_r, mv_rec_r)
        context_r_2, _ = self.align(f_ref_rec_r_2, mv_rec_r_2)
        context_r_3, _ = self.align(f_ref_rec_r_3, mv_rec_r_3)

        context_1 = self.conv1_1(torch.cat([context_l_1 * feature_mul_l, context_r_1 * feature_mul_r], dim=1))
        context_2 = self.conv1_2(torch.cat([context_l_2 * feature_mul_l_2, context_r_2 * feature_mul_r_2], dim=1))
        context_3 = self.conv1_3(torch.cat([context_l_3 * feature_mul_l_3, context_r_3 * feature_mul_r_3], dim=1))

        context3_up = self.conv3_up(context_3)
        context3_up = self.res_block3_up(context3_up)
        context3_out = self.conv3_out(context_3)
        context3_out = self.res_block3_out(context3_out)
        context2_up = self.conv2_up(torch.cat((context3_up, context_2), dim=1))
        context2_up = self.res_block2_up(context2_up)
        context2_out = self.conv2_out(torch.cat((context3_up, context_2), dim=1))
        context2_out = self.res_block2_out(context2_out)
        context1_out = self.conv1_out(torch.cat((context2_up, context_1), dim=1))
        context1_out = self.res_block1_out(context1_out)
        context1 = context_1 + context1_out
        context2 = context_2 + context2_out
        context3 = context_3 + context3_out
        return context1, context2, context3


class EncoderResNetWCond_3Layer(nn.Module): #ok
    r"""Encoder 3 down sample layer
    """

    def __init__(self, c_in=64, c_mid=96, c_out=128, bottleneck=True, res_num=1):
        super().__init__()
        self.c_mid = c_mid
        self.c_out = c_out
        self.down1 = ResidualBlockWithStride(c_in, c_mid, stride=2)
        self.res1 = ResidualBlock(c_mid, c_mid)
        self.down2 = ResidualBlockWithStride(c_mid, c_mid, stride=2)
        self.res2 = ResidualBlock(c_mid, c_mid)
        self.down3 = ResidualBlockWithStride(c_mid, c_mid, stride=2)
        self.res3 = ResidualBlock(c_mid, c_mid)
        self.down4 = conv3x3(c_mid, c_out, stride=2)
    def forward(self, x):
        x = self.down1(x)
        x = self.res1(x)
        x = self.down2(x)
        x = self.res2(x)
        x = self.down3(x)
        x = self.res3(x)
        x = self.down4(x)
        return x


class DecoderResNetGCond_3Layer(nn.Module): #ok
    r"""Decoder with 3 upsample
    """

    def __init__(self, c_in=128, c_out=64, bottleneck=False, res_num=2):
        super().__init__()
        c_mid = c_in
        self.res1 = ResidualBlock(c_in, c_mid)
        self.up1 = ResidualBlockUpsample(c_mid, c_mid, 2)
        self.res2 = ResidualBlock(c_mid, c_mid)
        self.up2 = ResidualBlockUpsample(c_mid, c_mid, 2)
        self.res3 = ResidualBlock(c_mid, c_mid)
        self.up3 = ResidualBlockUpsample(c_mid, c_mid, 2)
        self.res4 = ResidualBlock(c_mid, c_mid)
        self.up4 = subpel_conv1x1(c_mid, c_out, 2)

    def forward(self, x):
        x = self.res1(x)
        x = self.up1(x)
        x = self.res2(x)
        x = self.up2(x)
        x = self.res3(x)
        x = self.up3(x)
        x = self.res4(x)
        x, mul_pixel = self.up4(x).split([2, 3], dim=1)
        return x, mul_pixel
    
class Scaling_Net(nn.Module):
    def __init__(self, out_channel_lst):
        super(Scaling_Net,self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(1,64),
            nn.ReLU(),
            nn.Linear(64,sum(out_channel_lst))
        )
        self.initialize_weights(self.layer) # 0
        self.o_lst = out_channel_lst

    def initialize_weights(self,layers):
        pass
        for layer in layers:
            if isinstance(layer,nn.Linear):
                # nn.init.kaiming_normal_(layer.weight)
                # nn.init.constant_(layer.weight,0)
                nn.init.constant_(layer.bias, 0)

    def forward(self,x):
        x = torch.exp(self.layer(x)).contiguous().view(1,-1,1,1)
        return x.split(self.o_lst, dim=1)


class EncoderResNet_3Layer_Svr(EncoderResNet_3Layer):
    def forward(self, x, context1, context2, context3, scalars=None):
        if scalars is None:
            return super().forward(x, context1, context2, context3)
        scalars = scalars.split([self.c_in, self.c_in, self.c_out], dim=1)

        feature = self.conv1(torch.cat([x, context1], dim=1))
        feature = self.gdn1(feature)
        feature = feature * scalars[0]
        feature = self.res1(torch.cat([feature, context2], dim=1))
        feature = self.conv2(feature)
        feature = self.gdn2(feature)
        feature = feature * scalars[1]
        feature = self.res2(torch.cat([feature, context3], dim=1))
        feature = self.conv3(feature)
        feature = self.gdn3(feature)
        feature = self.conv4(feature)
        feature = feature * scalars[2]
        return feature


class DecoderResNet_3Layer_Svr(DecoderResNet_3Layer):
    def forward(self, feature, context2, context3, scalars=None):
        if scalars is None:
            return super().forward(feature, context2, context3)
        scalars = scalars.split([128, 64], dim=1)
        feature = feature * scalars[0]
        feature = self.up1(feature)
        feature = self.gdn1(feature)
        feature = self.up2(feature)
        feature = self.gdn2(feature)
        feature = self.res1(torch.cat([feature, context3], dim=1))
        feature = self.up3(feature)
        feature = self.gdn3(feature)
        feature = feature * scalars[1]
        feature = self.res2(torch.cat([feature, context2], dim=1))
        feature = self.up4(feature)
        return feature

class EncoderResNetWCond_3Layer_Svr(EncoderResNetWCond_3Layer):
    r"""Encoder 3 down sample layer
    """
    def forward(self, x, scalars=None):
        if scalars is None:
            return super().forward(x)
        scalars = scalars.split([self.c_mid, self.c_out], dim=1)
        x = self.down1(x)
        x = self.res1(x)
        x = x * scalars[0]
        x = self.down2(x)
        x = self.res2(x)
        x = self.down3(x)
        x = self.res3(x)
        x = self.down4(x)
        x = x * scalars[1]
        return x

class DecoderResNetGCond_3Layer_Svr(DecoderResNetGCond_3Layer):
    r"""Decoder with 3 upsample
    """
    def forward(self, x, scalars=None):
        if scalars is None:
            return super().forward(x)
        scalars = scalars.split([96, 96], dim=1)
        x = x * scalars[0]
        x = self.res1(x)
        x = self.up1(x)
        x = self.res2(x)
        x = self.up2(x)
        x = self.res3(x)
        x = self.up3(x)
        x = x * scalars[1]
        x = self.res4(x)
        x, mul_pixel = self.up4(x).split([2, 3], dim=1)
        return x, mul_pixel
    
class ResTPriorFusion(nn.Module):
    def __init__(self, in_ch=128, out_ch=128, inplace=False):
        super().__init__()
        self.combine = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 1)
        )

    def forward(self, x1, x2):
        if x1 is None:
            if x2 is None:
                return None
            else:
                x1 = x2
        else:
            if x2 is None:
                x2 = x1
        x = self.combine(x1/2 + x2/2)
        return x
    
    
class EEV(nn.Module):
    def __init__(self):
        super().__init__()
        self.mv_rec = MotionRec() 
        self.mv_choose = MotionChoose()
        self.feature_extractor = FeatureExtractor(channel=64)
        self.feature_extractor_X = FeatureExtractor(channel=64)
        self.feature_adaptor_I = FeatExtractor(c_in=3, c_out=64, stride=1)
        self.feature_adaptor_X = FeatExtractor(c_in=3, c_out=64, stride=1)
        self.feature_adaptor_P = FeatExtractor(c_in=64, c_out=64, stride=1, ks=1)
        self.feature_rec = FeatRec(c_in=128, N=64, c_out=3, ks=3)
        self.mv_encoder = EncoderResNetWCond_3Layer_Svr(c_in=4, c_mid=128, c_out=96)
        self.mv_decoder = DecoderResNetGCond_3Layer_Svr(c_in=96, c_out=5)
        self.mv_entropy = MvConditionHyperPriorEntropy(c_in=96, c_prior=96)
        self.align = Align2D()
        self.motion_compensation = ContextRefine(c_in=64, N=64)
        self.r_encoder = EncoderResNet_3Layer_Svr(c_in=64, c_out=128)
        self.r_decoder = DecoderResNet_3Layer_Svr(c_in=128, c_out=64)
        self.rsp_encoder = EncoderPrior_3Layer(c_in=64, c_out=128)
        self.r_entropy = ResConditionHyperPriorEntropy(c_in=128, c_prior=128)
        self.scalar = Scaling_Net(out_channel_lst=[224, 192, 192, 256, 192, 320])
        self.res_tprior_fusion = ResTPriorFusion()
        self.optic_flow = ME_Spynet()

    def process_infos(self, x, infos, ref_l=-1, ref_r=-1):
        x_ref_rec_l = infos[str(ref_l)]['x_ref_rec'] if 'x_ref_rec' in infos[str(ref_l)] else x
        f_ref_rec_l = infos[str(ref_l)]['f_ref_rec'] if 'f_ref_rec' in infos[str(ref_l)] else None
        mv_tprior_ll = infos[str(ref_l)]['mv_tprior_l'] if 'mv_tprior_l' in infos[str(ref_l)] else None
        mv_tprior_lr = infos[str(ref_l)]['mv_tprior_r'] if 'mv_tprior_r' in infos[str(ref_l)] else None
        res_tprior_l = infos[str(ref_l)]['res_tprior'] if 'res_tprior' in infos[str(ref_l)] else None
        
        x_ref_rec_r = infos[str(ref_r)]['x_ref_rec'] if 'x_ref_rec' in infos[str(ref_r)] else x
        f_ref_rec_r = infos[str(ref_r)]['f_ref_rec'] if 'f_ref_rec' in infos[str(ref_r)] else None
        mv_tprior_rl = infos[str(ref_r)]['mv_tprior_l'] if 'mv_tprior_l' in infos[str(ref_r)] else None
        mv_tprior_rr = infos[str(ref_r)]['mv_tprior_r'] if 'mv_tprior_r' in infos[str(ref_r)] else None
        res_tprior_r = infos[str(ref_r)]['res_tprior'] if 'res_tprior' in infos[str(ref_r)] else None
        return x_ref_rec_l, f_ref_rec_l, mv_tprior_ll, mv_tprior_lr, res_tprior_l, \
                x_ref_rec_r, f_ref_rec_r, mv_tprior_rl, mv_tprior_rr, res_tprior_r


    def process_vr(self, lambda_val):
        # generate scale vector
        lambda_val = torch.tensor(lambda_val, device=next(self.scalar.parameters()).device).view(1,1)
        mv_enc_s, mv_dec_s, mv_prior_s, r_enc_s, r_dec_s, r_prior_s = self.scalar(lambda_val)
        r_prior_s = r_prior_s.split([64, 128, 128], dim=1)
        mv_prior_s = mv_prior_s.chunk(2, dim=1)
        return mv_enc_s, mv_dec_s, mv_prior_s, r_enc_s, r_dec_s, r_prior_s


    def forward(self, x, infos={}, idx=-1, ref_l=-1, ref_r=-1, ec_mode='forward'):
        mv_enc_s, mv_dec_s, mv_prior_s, r_enc_s, r_dec_s, r_prior_s = self.process_vr(infos[str(idx)]['lambda'])
        x_ref_rec_l, f_ref_rec_l, mv_tprior_ll, mv_tprior_lr, res_tprior_l, \
                x_ref_rec_r, f_ref_rec_r, mv_tprior_rl, mv_tprior_rr, res_tprior_r = self.process_infos(
            x, infos, ref_l, ref_r)
        # mv compress
        est_mv_l = self.optic_flow(x, x_ref_rec_l)
        est_mv_ref_l = self.optic_flow(x_ref_rec_r, x_ref_rec_l) * 0.5
        mv_y_l = self.mv_encoder(torch.cat([est_mv_l, est_mv_ref_l], dim=1),
                                    scalars=mv_enc_s)

        mv_tprior_ll = mv_tprior_ll * mv_prior_s[0] if mv_tprior_ll is not None else None
        mv_tprior_lr = mv_tprior_lr * mv_prior_s[0] if mv_tprior_lr is not None else None
        y_likelihoods_l, z_likelihoods_l, mv_latent_hat_l = self.mv_entropy(
            mv_y_l, mv_tprior_ll, mv_tprior_lr)
        mv_rate_l = -(y_likelihoods_l.log2().sum() + z_likelihoods_l.log2().sum())
        mv_pred_l, pexel_mul_l = self.mv_decoder(mv_latent_hat_l, 
                                         scalars=mv_dec_s)
        pexel_mul_l = torch.sigmoid(pexel_mul_l)
        mv_rec_l = self.mv_rec(torch.cat([mv_pred_l, est_mv_ref_l], dim=1))
        feature_mul_l = self.mv_choose(pexel_mul_l)
        x_pred_l, _ = self.align(x_ref_rec_l, mv_rec_l)

        # mv compress
        est_mv_r = self.optic_flow(x, x_ref_rec_r)
        est_mv_ref_r = self.optic_flow(x_ref_rec_l, x_ref_rec_r) * 0.5
        mv_y_r = self.mv_encoder(torch.cat([est_mv_r, est_mv_ref_r], dim=1),
                                    scalars=mv_enc_s)

        mv_tprior_rl = mv_tprior_rl * mv_prior_s[0] if mv_tprior_rl is not None else None
        mv_tprior_rr = mv_tprior_rr * mv_prior_s[0] if mv_tprior_rr is not None else None
        y_likelihoods_r, z_likelihoods_r, mv_latent_hat_r = self.mv_entropy(
            mv_y_r, mv_tprior_rr, mv_tprior_rl)
        mv_rate_r = -(y_likelihoods_r.log2().sum() + z_likelihoods_r.log2().sum())
        mv_pred_r, pexel_mul_r = self.mv_decoder(mv_latent_hat_r, 
                                         scalars=mv_dec_s)
        pexel_mul_r = torch.sigmoid(pexel_mul_r)
        mv_rec_r = self.mv_rec(torch.cat([mv_pred_r, est_mv_ref_r], dim=1))
        feature_mul_r = self.mv_choose(pexel_mul_r)
        x_pred_r, _ = self.align(x_ref_rec_r, mv_rec_r)
        x_pred = x_pred_l * pexel_mul_l + x_pred_r * pexel_mul_r

        # align
        # left
        # extract feature
        if f_ref_rec_l is None:
            f_ref_rec_l = self.feature_adaptor_I(x_ref_rec_l)  # first I frame
        else:
            f_ref_rec_l = self.feature_adaptor_P(f_ref_rec_l)  # other P frames
        f_ref_rec_l, f_ref_rec_l_2, f_ref_rec_l_3 = self.feature_extractor(f_ref_rec_l)
        # right
        # extract feature
        if f_ref_rec_r is None:
            f_ref_rec_r = self.feature_adaptor_I(x_ref_rec_r)  # first I frame
        else:
            f_ref_rec_r = self.feature_adaptor_P(f_ref_rec_r)  # other P frames
        f_ref_rec_r, f_ref_rec_r_2, f_ref_rec_r_3 = self.feature_extractor(f_ref_rec_r)

        f_pred, f_pred_2, f_pred_3 = self.motion_compensation(f_ref_rec_l, f_ref_rec_l_2, f_ref_rec_l_3, \
                                          f_ref_rec_r, f_ref_rec_r_2, f_ref_rec_r_3, \
                                            mv_rec_l, mv_rec_r, feature_mul_l, feature_mul_r)
        
        # residual compress
        f_cur, f_cur_2, f_cur_3 = self.feature_extractor_X(self.feature_adaptor_X(x))
        res_latent = self.r_encoder(f_cur, f_pred, f_pred_2, f_pred_3, 
                                    scalars=r_enc_s)
        # residual prior: last residual (temporal) and predict feature (structure)
        res_tprior_l = res_tprior_l * r_prior_s[1] if res_tprior_l is not None else res_tprior_l
        res_tprior_r = res_tprior_r * r_prior_s[1] if res_tprior_r is not None else res_tprior_r
        res_tprior = self.res_tprior_fusion(res_tprior_l, res_tprior_r)
        
        y_likelihoods, z_likelihoods, res_latent_hat = self.r_entropy(
            res_latent, res_tprior, f_pred * r_prior_s[0], f_pred_2 * r_prior_s[0], f_pred_3 * r_prior_s[0])
        
        res_rate = -(y_likelihoods.log2().sum() + z_likelihoods.log2().sum())
        f_res_rec = self.r_decoder(res_latent_hat, f_pred_2, f_pred_3, r_dec_s)
        # reconstruct
        x_rec, f_rec = self.feature_rec(
            torch.cat([f_res_rec, f_pred], dim=1))
        # Update info
        
        infos[str(idx)].update({
            'mv_tprior_l': mv_latent_hat_l * mv_prior_s[-1],
            'mv_tprior_r': mv_latent_hat_r * mv_prior_s[-1],
            'res_tprior': res_latent_hat * r_prior_s[-1],
            'f_ref_rec': f_rec,
            'stream': self.r_entropy.flush() if ec_mode == 'encode' else None
        })
        return x_rec, {'mv': mv_rate_l + mv_rate_r, 'res': res_rate}, infos