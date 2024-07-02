import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import sys
import cv2
import torch
import numpy as np

import logging

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        print('For PyTorch <= 1.0, tensorboardX should be installed')
        sys.exit(1)

from torchvision.utils import save_image
from dataload.type_convert import rgb_to_yuv


class Singleton(object):
    def __init__(self, cls):
        self._cls = cls
        self._instance = {}

    def __call__(self, *args, **kwargs):
        if self._cls not in self._instance:
            self._instance[self._cls] = self._cls(*args, **kwargs)
        return self._instance[self._cls]


@Singleton
class Logger(object):
    def __init__(self,
                 root_path='./runs/',
                 logger_name='train',
                 level=logging.INFO,
                 toscreen=False,
                 tofile=True):
        self._logger = logging.getLogger(logger_name)
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
            datefmt='%y-%m-%d %H:%M:%S')
        self._logger.setLevel(level)
        self.root_path = root_path
        self.logger_name = logger_name
        if tofile:
            os.makedirs(root_path, exist_ok=True)
            log_file = os.path.join(root_path, logger_name + '.log')
            fh = logging.FileHandler(log_file, mode='w')
            fh.setFormatter(formatter)
            self._logger.addHandler(fh)
        if toscreen:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            self._logger.addHandler(sh)

    def get_logger(self):
        return self._logger

    def be_quiet(self):
        self._logger.setLevel(logging.WARNING)
    
    def get_path(self):
        return self.root_path
        

@Singleton
class EvalState(object):
    def __init__(self, is_eval=False):
        self._is_eval = is_eval
    
    def get_eval(self):
        return self._is_eval


@Singleton
class SummaryWriterSingleton(SummaryWriter):
    pass


@Singleton
class ImageSaver():
    def __init__(self, save_path='./runs/', prefix=''):
        self.save_path = save_path
        self.prefix = prefix
        self.check = True
    
    def viz_flow(self, flow):
        h, w = flow.shape[:2]
        hsv = np.zeros((h,w,3), np.uint8)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv[..., 2] = 255
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr

    def save_flow(self, flow, path):
        path = os.path.join(self.save_path, path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        flow = flow.detach().cpu().numpy()
        flow = np.transpose(flow, (1,2,0))
        flow = self.viz_flow(flow)
        cv2.imwrite(path, flow)

    def save_rgb2yuv(self, img, yuv_type, path=None, type='bilinear'):
        """Save image with `yuv` format.
        """
        if path is None:
            path = os.path.join(self.save_path, f'{self.prefix}.yuv')
        else:
            path = os.path.join(self.save_path, path)
        if path.endswith('.yuv') is False:
            path = path + '.yuv'
        assert img.dim() == 4 and img.shape[0] == 1 and img.shape[1] == 3

        yuv_type = int(yuv_type)
        max_v = 2**(yuv_type)-1
        if yuv_type == 8:
            torch_type = torch.uint8
            numpy_type = np.uint8
        else:
            torch_type = torch.int16
            numpy_type = np.uint16

        # trans to yuv
        yuv = torch.clamp(rgb_to_yuv(img.float()/255) * max_v, 0, max_v)
        y_part, u, v = yuv.split([1,1,1], dim=1)

        if type == 'nearest':
            u = u[:,:,::2,::2]
            v = v[:,:,::2,::2]
        else:
            b,_,h,w = u.shape
            u = torch.mean(u.reshape(b, 1, h//2, 2, w//2, 2), axis=(-1, -3))
            v = torch.mean(v.reshape(b, 1, h//2, 2, w//2, 2), axis=(-1, -3))

        uv_part = torch.cat([u,v], dim=1)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.check:
            if os.path.exists(path):
                os.remove(path)
            self.check = False
    
        f = open(path, 'a')
        # float => uint8
        y_part = y_part.to(dtype=torch_type)
        uv_part = uv_part.to(dtype=torch_type)
        y_part = y_part.contiguous().view(-1)
        u_part = uv_part[0, 0:1, :, :].contiguous().view(-1)
        v_part = uv_part[0, 1:, :, :].contiguous().view(-1)
        # CUDA tensor => ndarray
        y_part = y_part.cpu().numpy()
        u_part = u_part.cpu().numpy()
        v_part = v_part.cpu().numpy()
        yuv_arr = np.zeros(len(y_part)+len(u_part)+len(v_part), numpy_type)
        yuv_arr[:len(y_part)] = y_part
        yuv_arr[len(y_part):len(y_part)+len(u_part)] = u_part
        yuv_arr[len(y_part)+len(u_part):len(y_part)+len(u_part)+len(v_part)] = v_part
        # save to file
        yuv_arr.tofile(f)
        f.close()
        
    def save(self, img, name, yuv_type=0, feature=False, flow=False):
        path = os.path.join(self.save_path, f'{self.prefix}-{name}')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if flow:
            self.save_flow(img, path)
        elif yuv_type != 0:
            self.save_rgb2yuv(img, yuv_type, path)
        elif feature:
            save_image(img, path, normalize=True)
        else:
            save_image(img, path, normalize=True, range=(0, 255))


def get_tb_logger() -> SummaryWriter:
    return SummaryWriterSingleton(log_dir='./fake_path/')


def get_logger() -> Logger:
    return Logger().get_logger()


def get_img_saver() -> ImageSaver:
    return ImageSaver()


def get_logger_path():
    return Logger().get_path()


def init_loggers(dir_path, logger_name, use_tb_logger, suffix=None, screen=True):
    if suffix is not None:
        logger = Logger(dir_path, '-'.join([logger_name, suffix]), toscreen=screen).get_logger()
    else:
        logger = Logger(dir_path, logger_name, toscreen=screen).get_logger()

    tb_logger = None
    if use_tb_logger:
        tb_logger = SummaryWriterSingleton(os.path.join(dir_path, 'tb_logs'))

    ImageSaver(os.path.join(dir_path, 'imgs'), logger_name)
    # print(train_url, opt.train.logger.logger_name)
    return logger, tb_logger


def is_eval():
    return EvalState().get_eval()


def init_state(is_eval):
    EvalState(is_eval)


def set_quiet():
    Logger().be_quiet()

