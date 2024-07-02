import os
import argparse
import torch
import random
import math
import datetime
import numpy as np
from models.Network import EEV
from models.Network_I import IFrame
from torch.utils.data import DataLoader
from dataload.dataset import VideoDataset
from tqdm import tqdm
import imageio
from utils.metrics import psnr as psnr_fn
from utils.metrics import MSSSIMLossRGB2YUV
import torch.nn.functional as F
from models.utils import get_code_seq
from dataload.type_convert import rgb_to_yuv, yuv_to_rgb
from models.utils import denormalize_img

metric_list = ['mse', 'ms-ssim']
parser = argparse.ArgumentParser(description='DMVC evaluation')

parser.add_argument('--pretrain', default = '', help='Load pretrain model')
parser.add_argument('--img_dir', default = '')
parser.add_argument('--eval_lambda', default = 256, type = int, help = '[256, 512, 1024, 2048] for MSE, [8, 16, 32, 64] for MS-SSIM')
parser.add_argument('--metric', default = 'mse', choices = metric_list, help = 'mse or ms-ssim')
parser.add_argument('--gop_size', default = '0', type = int, help = 'The length of the gop')

parser.add_argument('--src',              type=str, required=True)
parser.add_argument('--seq_name',              type=str, required=True)
parser.add_argument('--width',            type=int, required=True)
parser.add_argument('--height',           type=int, required=True)
parser.add_argument('--device',           type=str, choices=["cuda", "cpu"], default="cuda")
parser.add_argument('--num_frames',       type=int, default=-1, help="number of frame to test")

parser.add_argument('--intra_period',     type=int, default=16, help="intra period")
parser.add_argument('--lambda_i',         type=int, default=0, choices=[0, 1, 2, 3])

args = parser.parse_args()

test_dataset = VideoDataset(args.src, args.num_frames, args.intra_period, args.gop_size, args.width, args.height)  
test_loader = DataLoader(dataset = test_dataset, shuffle = False, num_workers = 1, batch_size = 1)
ssim_fn = MSSSIMLossRGB2YUV(6)#$MS_SSIM(data_range=1., reduction="none").to(args.device)


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def get_padding_size(height, width, p=64):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    # padding_left = (new_w - width) // 2
    padding_left = 0
    padding_right = new_w - width - padding_left
    # padding_top = (new_h - height) // 2
    padding_top = 0
    padding_bottom = new_h - height - padding_top
    return padding_left, padding_right, padding_top, padding_bottom

def eval_model(net, intra_net):
    print("Evaluating...")
    net.eval()
    intra_net.eval()

    i_quality = [63, 47, 31,15]
    infos = {}
    infos_idx = []
    img_rate = 0
    mv_rate = 0
    res_rate = 0
    w, h = args.width, args.height
    batch_size = 1
    numel = batch_size * h * w
    padding_l, padding_r, padding_t, padding_b = get_padding_size(h, w, 64)
    seq_idx, ref_idx, layers_idx = get_code_seq(args.intra_period) 
    layers_sum = layers_idx[1]
    with torch.no_grad():
        x_gop = []
        x_yuv_gop = []
        for idx, (pair, x, x_init_yuv) in tqdm(enumerate(test_loader), total=len(test_loader)):
            x_gop.append(x)
            x_yuv_gop.append(x_init_yuv)
            if idx == 0:
                x_padded = F.pad(
                    x,
                    (padding_l, padding_r, padding_t, padding_b),
                    mode="replicate",
                )
                yuv_padded = F.pad(
                    x_init_yuv,
                    (padding_l, padding_r, padding_t, padding_b),
                    mode="replicate",
                )
                x_padded = x_padded.to(args.device)
                yuv_padded = yuv_padded.to(args.device)
                img_init = x.to(args.device)
                #self.frame_type = 'i'
                infos = {} # clear buffer
                infos[str(args.intra_period)] = {}
                infos_idx = [] 
                result = intra_net(yuv_padded, i_quality[args.lambda_i])
                x_rec = result["x_hat"]
                bit = result["bit"]
                bit = {'img': bit}
                x_rec = torch.clamp(yuv_to_rgb(x_rec), 0, 1)

                infos[str(args.intra_period)].update({
                    'x_ref_rec': x_rec,
                    'x_ref_ori': x_padded
                })
                x_rec = x_rec[:, :, :h, :w]
                psnr_yuv = psnr_fn(x_rec, img_init)
                x_rec = denormalize_img(x_rec)
            elif idx % args.intra_period == 0:
                for test_iter in range(0, args.intra_period):
                    x_padded = F.pad(
                        x_gop[seq_idx[test_iter]],
                        (padding_l, padding_r, padding_t, padding_b),
                        mode="replicate",
                    )
                    yuv_padded = F.pad(
                        x_yuv_gop[seq_idx[test_iter]],
                        (padding_l, padding_r, padding_t, padding_b),
                        mode="replicate",
                    )
                    x_padded = x_padded.to(args.device)
                    yuv_padded = yuv_padded.to(args.device)
                    img_init = x_gop[seq_idx[test_iter]].to(args.device)
                    #self.intra_period = 16
                    if test_iter == 0:
                        #self.frame_type = 'i'
                        tmp = infos[str(args.intra_period)]
                        infos = {} # clear buffer
                        infos['0'] = tmp
                        infos[str(args.intra_period)] = {}
                        infos_idx = []
                        info_idx = args.intra_period

                        #self.x_rec, bit, self.infos[str(info_idx)] = \
                        #    self.frames[self.frame_type](self.x, self.infos[str(info_idx)])

                        result = intra_net(yuv_padded, i_quality[args.lambda_i])
                        x_rec = result["x_hat"]
                        bit = result["bit"]
                        bit = {'img': bit}
                        x_rec = torch.clamp(yuv_to_rgb(x_rec), 0, 1)
                        
                        infos[str(info_idx)].update({
                            'x_ref_rec': x_rec,
                            'x_ref_ori': x_padded
                        })
                        x_rec = x_rec[:, :, :h, :w]

                    else:
                        #self.frame_type = 'p'
                        info_idx = seq_idx[test_iter]
                        ref_l = info_idx - ref_idx[info_idx]
                        ref_r = info_idx + ref_idx[info_idx]
                        infos[str(info_idx)] = {}
                        if args.lambda_i == 0:
                            # 2048
                            if layers_idx[info_idx] == layers_sum - 4:
                                infos[str(info_idx)] = {'lambda': 1.4}
                            elif layers_idx[info_idx] == layers_sum - 3:
                                infos[str(info_idx)] = {'lambda': 1.4}
                            elif layers_idx[info_idx] == layers_sum - 2:
                                infos[str(info_idx)] = {'lambda': 0.6938}
                            elif layers_idx[info_idx] == layers_sum - 1:
                                infos[str(info_idx)] = {'lambda': 0.5091}
                            else:
                                infos[str(info_idx)] = {'lambda': 0.1945}
                        elif args.lambda_i == 1:
                            # 512
                            if layers_idx[info_idx] == layers_sum - 4:
                                infos[str(info_idx)] = {'lambda': 1.4}
                            elif layers_idx[info_idx] == layers_sum - 3:
                                infos[str(info_idx)] = {'lambda': 1.4}
                            elif layers_idx[info_idx] == layers_sum - 2:
                                infos[str(info_idx)] = {'lambda': 0.4639}
                            elif layers_idx[info_idx] == layers_sum - 1:
                                infos[str(info_idx)] = {'lambda': 0.3488}
                            else:
                                infos[str(info_idx)] = {'lambda': 0.1368}
                        elif args.lambda_i == 2 or args.lambda_i == 3:
                            # 64/256
                            if layers_idx[info_idx] == layers_sum - 4:
                                infos[str(info_idx)] = {'lambda': 1.4}
                            elif layers_idx[info_idx] == layers_sum - 3:
                                infos[str(info_idx)] = {'lambda': 1.4}
                            elif layers_idx[info_idx] == layers_sum - 2:
                                infos[str(info_idx)] = {'lambda': 0.3472}
                            elif layers_idx[info_idx] == layers_sum - 1:
                                infos[str(info_idx)] = {'lambda': 0.2756}
                            else:
                                infos[str(info_idx)] = {'lambda': 0.1137}
                            
                        while len(infos_idx) > layers_idx[info_idx]:
                            infos.pop(str(infos_idx[-1]))
                            infos_idx.pop()

                        x_rec, bit, infos = \
                            net(x_padded, infos, idx=info_idx, ref_l=ref_l, ref_r=ref_r)
                        infos[str(info_idx)].update({
                            'idx': info_idx,
                            'x_ref_rec': x_rec,
                            'x_ref_ori': x_padded
                        })
                        infos_idx.append(info_idx) # clear buffer
                        x_rec = x_rec[:, :, :h, :w]
                    mv_rate = (bit['mv'] / numel).item() if 'mv' in bit else 0
                    res_rate = (bit['res'] / numel).item() if 'res' in bit else 0
                    img_rate = (bit['img'] / numel).item() if 'img' in bit else 0
                    rate = mv_rate + img_rate + res_rate
                    psnr_yuv = psnr_fn(x_rec, img_init)
                    print(rate, psnr_yuv)
                x_gop = [0]
                x_yuv_gop = [0]

def check_dir_exist(check_dir):
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)

def main():
    print(args)

    model = EEV()
    model.cuda()

    model_i = IFrame()
    model_i.cuda()

    if args.pretrain != '':
        print('Load the model from {}'.format(args.pretrain))
        pretrained_dict_p = torch.load(args.pretrain)['p']
        pretrained_dict_p2 = {}
        for k, v in pretrained_dict_p.items():
            for k1, v1 in pretrained_dict_p[k].items():
                pretrained_dict_p2[k + '.' + k1] = v1
        model_dict = model.state_dict()
        pretrained_dict_p2 = {k: v for k, v in pretrained_dict_p2.items() if k in model_dict}
        #for k, v in pretrained_dict_i2.items():
        #    print(k)
        model_dict.update(pretrained_dict_p2)
        model.load_state_dict(model_dict, strict=True)

        pretrained_dict_i = torch.load(args.pretrain)['i']
        pretrained_dict_i2 = {}
        for k, v in pretrained_dict_i.items():
            for k1, v1 in pretrained_dict_i[k].items():
                pretrained_dict_i2[k + '.' + k1] = v1
        model_dict = model_i.state_dict()
        pretrained_dict_i2 = {k: v for k, v in pretrained_dict_i2.items() if k in model_dict}
        model_dict.update(pretrained_dict_i2)
        model_i.load_state_dict(model_dict, strict=True)

        '''
        for k, v in pretrained_dict.items():
            print(k)
        '''
    

    eval_model(model, model_i)

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    main()
