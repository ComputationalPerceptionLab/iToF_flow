import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backwarp_tenGrid = {}


def warp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1).to(device)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)

    #生成mask
    mask = torch.ones(tenInput.size()).to(dtype=tenInput.dtype, device=tenInput.device)
    mask = torch.nn.functional.grid_sample(input=mask, grid=g, mode='bilinear', padding_mode='border', align_corners=True)
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output * mask, mask


# lightspeed in nanoseconds
constants = {'lightspeed': 0.299792458}



def correlation2depth(correlations1,correlations2,correlations3,correlations4, frequencies, eps=1e-6):
  frequencies = frequencies/1e3

  signx = torch.sign(correlations1 - correlations3)
  signx[signx == 0] = 1

  signy = torch.sign(correlations2 - correlations4)
  signy[signy == 0] = 1

  delta_phi = torch.atan2(correlations2 - correlations4 + signy * eps, correlations1 - correlations3+ signx * eps)
  delta_phi[delta_phi < 0] += 2 * np.pi

  tof_depth = constants['lightspeed'] / (4 * np.pi * frequencies) * delta_phi

  return tof_depth


