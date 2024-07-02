import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Align2D(nn.Module):
    r"""Align image with reference and flow.
    """

    def __init__(self):
        super(Align2D, self).__init__()

    def get_grid(self, flow, w, h):
        gridX, gridY = np.meshgrid(np.arange(w), np.arange(h))
        gridX = torch.tensor(gridX, requires_grad=False, device=flow.device)
        gridY = torch.tensor(gridY, requires_grad=False, device=flow.device)
        u = flow[:, 0]
        v = flow[:, 1]
        x = gridX.unsqueeze(0).expand_as(u).float() + u
        y = gridY.unsqueeze(0).expand_as(v).float() + v
        x = 2 * (x / (w - 1) - 0.5)
        y = 2 * (y / (h - 1) - 0.5)
        grid = torch.stack((x, y), dim=3)
        return grid

    def init_dcn(self):
        pass

    def forward(self, x, flow):
        h, w = x.shape[-2:]
        x_warp = F.grid_sample(x,
                               self.get_grid(flow, w, h,),
                               mode='bilinear',
                               padding_mode='border',
                               align_corners=True)

        return x_warp, flow

