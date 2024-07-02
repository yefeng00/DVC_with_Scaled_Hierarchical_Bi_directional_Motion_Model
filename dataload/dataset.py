import torch
import numpy as np
import random
from .type_convert import yuv_to_rgb, rgb_to_yuv
import torch.nn.functional as F
from torch.utils.data import Dataset as torchData

class VideoDataset(torchData):
    def __init__(self, root, num_frame, intra_period, gop_size, width, height, no_img=False):
        super(VideoDataset, self).__init__()
        self.root = root + '.yuv'
        self.width = width
        self.height = height
        self.num_frame = (num_frame // intra_period) * intra_period + 1
        self.intra_period = intra_period
        self.gop_size = gop_size
        self.no_img = no_img

    def __len__(self):
        return self.num_frame

    def __getitem__(self, idx):
        frame_idx = idx

        '''if not self.no_img:
            raw_path = f"{self.root}/im{str(frame_idx).zfill(3)}.png"
            img = to_tensor(imgloader(raw_path))
            return pair, img
        el'''
        if True:
            width = self.width
            height = self.height
            bits, data_ratio, data_type = 8, 1, 'uint8'
            #abs_path = '/home/yefeng/data/Class/rgb_crop/gbr/yuv420p/BasketballPass_384x192.yuv'#self.root #os.path.join(self.root, seq_info[0])
            yuv_file = open(self.root, "rb")
            yuv_file.seek(frame_idx * width * height//2*3* data_ratio)
            image = np.fromfile(yuv_file, data_type, width*height//2*3)#.to('cuda', non_blocking=True)

            dev = 2**(bits-8)
        
            img = torch.Tensor(image).float()
            y_part = img[:width*height].view(height, width).unsqueeze(0).unsqueeze(0)
            u_part = img[width*height:width*height+width*height//4].view(height//2, width//2).unsqueeze(0)
            v_part = img[width*height+width*height//4:width*height+width*height//2].view(height//2, width//2).unsqueeze(0)

            uv_part = torch.cat((u_part, v_part), dim=0).unsqueeze(0)
            # unsample uv_t
            uv_part = F.interpolate(uv_part, scale_factor=2)
            image0 =  torch.cat([y_part.float()/dev, uv_part.float()/dev], dim=1)
            image_yuv = image0.squeeze()/255.0
            #image_yuv = torch.clamp(rgb_to_yuv(image.float()/255)*255, 0, 255)/255
            image = torch.clamp(yuv_to_rgb(image0/255)*255, 0, 255).squeeze()/255
            #print(torch.clamp(rgb_to_yuv(image.float()/255.0)*255, 0, 255))
            #image = (image.squeeze().permute(1, 2, 0).cpu().numpy())#.astype(np.uint8)
            #print(image.shape)
            #print(image)
            #print(image/255.0)
            #print(image_yuv == image0/255)
            #image = (image.squeeze().clamp(0, 255).permute(1, 2, 0).cpu().numpy()).astype(np.uint8)


            #image = np.append(image, (width, height, bits))
            #x_hat_write = (image.squeeze().clamp(0, 255).permute(1, 2, 0).cpu().numpy()).astype(np.uint8)
            #image = torch.from_numpy(image)
            #image = (image.squeeze().clamp(0, 255).permute(1, 2, 0).cpu().numpy()).astype(np.uint8)
            #imageio.imwrite("/home/yefeng/B-CANF-main/result/reco_{}.png".format(idx), image)

            #raw_path = f"{self.root}/im{str(frame_idx).zfill(3)}.png"
            #img = to_tensor(imgloader(raw_path))
            #img = to_tensor(image)
            #print(img.shape)
            return 0, image, image_yuv
        else:
            return pair
