import os
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
import copy

from dataset_LLT_SF_json import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
# from prefetch_generator import BackgroundGenerator


device = torch.device("cuda")

log_path = 'train_log'

CMAP_JET_MASK = copy.copy(plt.cm.jet)
CMAP_JET_MASK.set_bad(color='black')

CMAP_VIRIDIS_MASK = copy.copy(plt.cm.viridis)
CMAP_VIRIDIS_MASK.set_bad(color='black')

CMAP_CW_MASK = copy.copy(plt.cm.coolwarm)
CMAP_CW_MASK.set_bad(color='black')

PLT_ERROR_CONFIG = {'cmap': CMAP_CW_MASK}
PLT_DEPTH_CONFIG = {'cmap': CMAP_VIRIDIS_MASK}
PLT_CORR_CONFIG = {'cmap': plt.get_cmap('gist_gray')}


def load_val_data(json_path):
    with open(json_path, 'r') as f:

        content = f.read()
        data = json.loads(content)
        img_00_path_array = data.get('img_00_path_array')
        img_11_path_array = data.get('img_11_path_array')
        img_22_path_array = data.get('img_22_path_array')
        img_33_path_array = data.get('img_33_path_array')

        img_40_path_array = data.get('img_40_path_array')

        img_40_gt_path_array = data.get('img_40_gt_path_array')
        img_41_gt_path_array = data.get('img_41_gt_path_array')
        img_42_gt_path_array = data.get('img_42_gt_path_array')
        img_43_gt_path_array = data.get('img_43_gt_path_array')

        img_00_LLT_path_array = data.get('img_00_LLT_path_array')
        img_01_LLT_path_array = data.get('img_01_LLT_path_array')
        img_02_LLT_path_array = data.get('img_02_LLT_path_array')
        img_03_LLT_path_array = data.get('img_03_LLT_path_array')

        frequency1 = data.get('freq')
        frequency2 = data.get('freq')

        for i in range(len(img_00_path_array)):
            img_00_path = img_00_path_array[i]
            img_11_path = img_11_path_array[i]
            img_22_path = img_22_path_array[i]
            img_33_path = img_33_path_array[i]

            img_40_path = img_40_path_array[i]

            img_40_gt_path = img_40_gt_path_array[i]
            img_41_gt_path = img_41_gt_path_array[i]
            img_42_gt_path = img_42_gt_path_array[i]
            img_43_gt_path = img_43_gt_path_array[i]

            img_00_LLT_path = img_00_LLT_path_array[i]
            img_01_LLT_path = img_01_LLT_path_array[i]
            img_02_LLT_path = img_02_LLT_path_array[i]
            img_03_LLT_path = img_03_LLT_path_array[i]


            img_00 = imageio.imread(os.path.join( img_00_path), format='HDR-FI')[0:448, 0:448, 0]
            img_11 = imageio.imread(os.path.join(img_11_path), format='HDR-FI')[0:448, 0:448, 0]
            img_22 = imageio.imread(os.path.join( img_22_path), format='HDR-FI')[0:448, 0:448, 0]
            img_33 = imageio.imread(os.path.join( img_33_path), format='HDR-FI')[0:448, 0:448, 0]
            
            img_40 = imageio.imread(os.path.join(img_40_path), format='HDR-FI')[0:448, 0:448, 0]

            img_40_gt = imageio.imread(os.path.join(img_40_gt_path), format='HDR-FI')[0:448, 0:448, 0]
            img_41_gt = imageio.imread(os.path.join(img_41_gt_path), format='HDR-FI')[0:448, 0:448, 0]
            img_42_gt = imageio.imread(os.path.join( img_42_gt_path), format='HDR-FI')[0:448, 0:448, 0]
            img_43_gt = imageio.imread(os.path.join( img_43_gt_path), format='HDR-FI')[0:448, 0:448, 0]

            img_00_LLT = imageio.imread(os.path.join(img_00_LLT_path), format='HDR-FI')[0:448, 0:448, 0]
            img_01_LLT = imageio.imread(os.path.join( img_01_LLT_path), format='HDR-FI')[0:448, 0:448, 0]
            img_02_LLT = imageio.imread(os.path.join( img_02_LLT_path), format='HDR-FI')[0:448, 0:448, 0]
            img_03_LLT = imageio.imread(os.path.join( img_03_LLT_path), format='HDR-FI')[0:448, 0:448, 0]
            
            max_value = 4000
            

            img_00[img_00 > max_value] = max_value
            img_11[img_11 > max_value] = max_value
            img_22[img_22 > max_value] = max_value
            img_33[img_33 > max_value] = max_value

            img_40[img_40 > max_value] = max_value

            img_40_gt[img_40_gt > max_value] = max_value
            img_41_gt[img_41_gt > max_value] = max_value
            img_42_gt[img_42_gt > max_value] = max_value
            img_43_gt[img_43_gt > max_value] = max_value

            img_00_LLT[img_00_LLT > max_value] = max_value
            img_01_LLT[img_01_LLT > max_value] = max_value
            img_02_LLT[img_02_LLT > max_value] = max_value
            img_03_LLT[img_03_LLT > max_value] = max_value



 
            img_00 = img_00 / max_value
            img_11 = img_11 / max_value
            img_22 = img_22 / max_value
            img_33 = img_33 / max_value

            img_40 = img_40 / max_value

            img_40_gt = img_40_gt / max_value
            img_41_gt = img_41_gt / max_value
            img_42_gt = img_42_gt / max_value
            img_43_gt = img_43_gt / max_value

            img_00_LLT = img_00_LLT / max_value
            img_01_LLT = img_01_LLT / max_value
            img_02_LLT = img_02_LLT / max_value
            img_03_LLT = img_03_LLT / max_value

            freq2 = np.full((448, 448), np.float(frequency2), dtype=np.float32)
            freq1 = np.full((448, 448), np.float(frequency1), dtype=np.float32)

            img_00 = torch.from_numpy(img_00.copy()).unsqueeze(0)
            img_11 = torch.from_numpy(img_11.copy()).unsqueeze(0)
            img_22 = torch.from_numpy(img_22.copy()).unsqueeze(0)
            img_33 = torch.from_numpy(img_33.copy()).unsqueeze(0)
            img_40 = torch.from_numpy(img_40.copy()).unsqueeze(0)

            img_40_gt = torch.from_numpy(img_40_gt.copy()).unsqueeze(0)
            img_41_gt = torch.from_numpy(img_41_gt.copy()).unsqueeze(0)
            img_42_gt = torch.from_numpy(img_42_gt.copy()).unsqueeze(0)
            img_43_gt = torch.from_numpy(img_43_gt.copy()).unsqueeze(0)

            img_00_LLT = torch.from_numpy(img_00_LLT.copy()).unsqueeze(0)
            img_01_LLT = torch.from_numpy(img_01_LLT.copy()).unsqueeze(0)
            img_02_LLT = torch.from_numpy(img_02_LLT.copy()).unsqueeze(0)
            img_03_LLT = torch.from_numpy(img_03_LLT.copy()).unsqueeze(0)

            freq2 = torch.from_numpy(freq2.copy()).unsqueeze(0)
            freq1 = torch.from_numpy(freq1.copy()).unsqueeze(0)

            res_tensor = torch.cat((img_00, img_11, img_22, img_33, img_40_gt, img_41_gt,img_42_gt,img_43_gt, freq1, img_01_LLT, img_02_LLT, img_03_LLT ), 0)

        return res_tensor






def plot_correlation_warp(img1_gt,warped_img1,img0_1, img2_gt, warped_img2,img0_2, img3_gt,warped_img3,img0_3,
                                            photo_bias1, photo_bias2, photo_bias3, gt_tof_depth,warped_tof_depth_refined, warped_tof_depth, warped_img1_unrefined, warped_img2_unrefined, warped_img3_unrefined, i,step=0,writer=None):
    fig = plt.figure(figsize=(25, 15))

    img1_gt = img1_gt.squeeze(0)
    warped_img1 = warped_img1.squeeze(0)
    img0_1 = img0_1.squeeze(0)
    img2_gt = img2_gt.squeeze(0)
    warped_img2 = warped_img2.squeeze(0)
    img0_2 = img0_2.squeeze(0)
    img3_gt = img3_gt.squeeze(0)
    warped_img3 = warped_img3.squeeze(0)
    img0_3 = img0_3.squeeze(0)
    photo_bias1 = photo_bias1.squeeze(0)
    photo_bias2 = photo_bias2.squeeze(0)
    photo_bias3 = photo_bias3.squeeze(0)
    gt_tof_depth = gt_tof_depth.squeeze(0)
    warped_tof_depth_refined = warped_tof_depth_refined.squeeze(0)
    warped_tof_depth = warped_tof_depth.squeeze(0)

    warped_img1_unrefined = warped_img1_unrefined.squeeze(0)
    warped_img2_unrefined = warped_img2_unrefined.squeeze(0)
    warped_img3_unrefined = warped_img3_unrefined.squeeze(0)

    

    fig = plt.figure(figsize=(25, 15))
    ax = plt.subplot(4,6,1)
    ax.imshow(img0_1 , **PLT_CORR_CONFIG)
    ax.set_title('img0_1')

    ax = plt.subplot(4,6,2)
    ax.imshow(warped_img1 , **PLT_CORR_CONFIG)
    ax.set_title('warped_img1')

    ax = plt.subplot(4,6,3)
    ax.imshow(img1_gt, **PLT_CORR_CONFIG)
    ax.set_title('img1_gt')

    ax = plt.subplot(4,6,4)
    error = warped_img1 - img1_gt
    m = np.max(np.abs(error))
    imsh = ax.imshow(np.abs(error), **PLT_ERROR_CONFIG, vmin=-m, vmax=m)
    ax.set_title('warped_img1 - img1_gt')
    fig.colorbar(imsh, ax=ax, shrink=0.5)

    ax = plt.subplot(4,6,5)
    error = img0_1 - warped_img1
    imsh = ax.imshow(np.abs(error), **PLT_ERROR_CONFIG, vmin=-m, vmax=m)
    ax.set_title('img0_1 - warped_img1')
    fig.colorbar(imsh, ax=ax, shrink=0.5)

    ax = plt.subplot(4,6,6)
    ax.imshow(photo_bias1 * 100, **PLT_CORR_CONFIG)
    ax.set_title('photo_bias1')

    ax = plt.subplot(4,6,7)
    ax.imshow(img0_2 , **PLT_CORR_CONFIG)
    ax.set_title('img0_2')

    ax = plt.subplot(4,6,8)
    ax.imshow(warped_img2 , **PLT_CORR_CONFIG)
    ax.set_title('warped_img2')

    ax = plt.subplot(4,6,9)
    ax.imshow(img2_gt, **PLT_CORR_CONFIG)
    ax.set_title('img2_gt')

    ax = plt.subplot(4,6,10)
    error = warped_img2 - img2_gt
    m = np.max(np.abs(error))
    imsh = ax.imshow(np.abs(error), **PLT_ERROR_CONFIG, vmin=-m, vmax=m)
    ax.set_title('warped_img2 - img2_gt')
    fig.colorbar(imsh, ax=ax, shrink=0.5)

    ax = plt.subplot(4,6,11)
    error = img0_2 - warped_img2
    imsh = ax.imshow(np.abs(error), **PLT_ERROR_CONFIG, vmin=-m, vmax=m)
    ax.set_title('img0_2 - warped_img2')
    fig.colorbar(imsh, ax=ax, shrink=0.5)

    ax = plt.subplot(4,6,12)
    ax.imshow(photo_bias2 * 100, **PLT_CORR_CONFIG)
    ax.set_title('photo_bias2')

    ax = plt.subplot(4,6,13)
    ax.imshow(img0_3 , **PLT_CORR_CONFIG)
    ax.set_title('img0_3')

    ax = plt.subplot(4,6,14)
    ax.imshow(warped_img3 , **PLT_CORR_CONFIG)
    ax.set_title('warped_img3')

    ax = plt.subplot(4,6,15)
    ax.imshow(img3_gt, **PLT_CORR_CONFIG)
    ax.set_title('img3_gt')

    ax = plt.subplot(4,6,16)
    error = warped_img3 - img3_gt
    m = np.max(np.abs(error))
    imsh = ax.imshow(np.abs(error), **PLT_ERROR_CONFIG, vmin=-m, vmax=m)
    ax.set_title('warped_img3 - img3_gt')
    fig.colorbar(imsh, ax=ax, shrink=0.5)

    ax = plt.subplot(4,6,17)
    error = img0_3 - warped_img3
    imsh = ax.imshow(np.abs(error), **PLT_ERROR_CONFIG, vmin=-m, vmax=m)
    ax.set_title('img0_3 - warped_img3')
    fig.colorbar(imsh, ax=ax, shrink=0.5)

    ax = plt.subplot(4,6,18)
    ax.imshow(photo_bias3 * 100, **PLT_CORR_CONFIG)
    ax.set_title('photo_bias3')

    ax = plt.subplot(4,6,19)
    ax.imshow(gt_tof_depth , **PLT_DEPTH_CONFIG)
    ax.set_title('gt_tof_depth')

    ax = plt.subplot(4,6,20)
    ax.imshow(warped_tof_depth , **PLT_DEPTH_CONFIG)
    ax.set_title('warped_tof_depth')

    ax = plt.subplot(4,6,21)
    ax.imshow(warped_tof_depth_refined , **PLT_DEPTH_CONFIG)
    ax.set_title('warped_tof_depth_refined')


    ax = plt.subplot(4,6,22)
    error = warped_tof_depth_refined - gt_tof_depth
    m = 0.5
    imsh = ax.imshow(np.abs(error), **PLT_ERROR_CONFIG, vmin=-m, vmax=m)
    ax.set_title('warped_tof_depth_refined - gt_tof_depth')
    fig.colorbar(imsh, ax=ax, shrink=0.5)


    fig.tight_layout()

    plt.savefig('evl_ours_visualization/' + str(i) + '.png')
    writer.add_figure(str(i) + '/figure', fig, step)
    plt.close()



writer_val = SummaryWriter('evl_ours_visualization')

from model.tof_cvpr_SF import Model
model = Model()
model.load_model_evl('./train_log/', -1)
print("load model.tof")

model.eval()
model.device()




json_path = 'val_data/evl_data.json'
data = load_val_data(json_path)
data = data.unsqueeze(0)
print(data.shape)  

seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

data_gpu = data.to(device, non_blocking=True) 
with torch.no_grad():
    warped_img1,warped_img2,warped_img3,info = model.update(data_gpu,  training=False)


    img0_gt = (data_gpu[:, 4:5].detach().cpu().numpy() )
    img1_gt = (data_gpu[:, 5:6].detach().cpu().numpy() )
    img2_gt = (data_gpu[:, 6:7].detach().cpu().numpy() )
    img3_gt = (data_gpu[:, 7:8].detach().cpu().numpy() )
    img0_1 = (data_gpu[:, 1:2].detach().cpu().numpy() )
    img0_2 = (data_gpu[:, 2:3].detach().cpu().numpy() )
    img0_3 = (data_gpu[:, 3:4].detach().cpu().numpy() )

    img0_1_0 = data_gpu[:, 9:10]
    img0_2_0 = data_gpu[:, 10:11]
    img0_3_0 = data_gpu[:, 11:12]


    photo_bias1 = info['photo_bias1'].detach().cpu().numpy()
    photo_bias2 = info['photo_bias2'].detach().cpu().numpy()
    photo_bias3 = info['photo_bias3'].detach().cpu().numpy()
    warped_img1 = warped_img1.detach().cpu().numpy()
    warped_img2 = warped_img2.detach().cpu().numpy()
    warped_img3 = warped_img3.detach().cpu().numpy()

    warped_img1_unrefined = info['warped_img1'].detach().cpu().numpy()
    warped_img2_unrefined = info['warped_img2'].detach().cpu().numpy()
    warped_img3_unrefined = info['warped_img3'].detach().cpu().numpy()


    gt_tof_depth = info['gt_tof_depth'].detach().cpu().numpy()
    warped_tof_depth = info['warped_tof_depth'].detach().cpu().numpy()
    warped_tof_depth_refined = info['warped_tof_depth_refined'].detach().cpu().numpy()


    plot_correlation_warp(img1_gt[0],warped_img1[0],img0_1[0], img2_gt[0], warped_img2[0],img0_2[0], img3_gt[0],warped_img3[0],img0_3[0],photo_bias1[0], photo_bias2[0], photo_bias3[0],gt_tof_depth[0], warped_tof_depth_refined[0], warped_tof_depth[0],warped_img1_unrefined[0], warped_img2_unrefined[0], warped_img3_unrefined[0], 0, 0, writer_val)