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

from model.tof_cvpr_SF import Model
from dataset_LLT_SF_json import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

device = torch.device("cuda")

log_path = 'train_log'

#tensorboard只能显示标准png图片，因此需要绘制之后再保存
CMAP_JET_MASK = copy.copy(plt.cm.jet)
CMAP_JET_MASK.set_bad(color='black')

CMAP_VIRIDIS_MASK = copy.copy(plt.cm.viridis)
CMAP_VIRIDIS_MASK.set_bad(color='black')

CMAP_CW_MASK = copy.copy(plt.cm.coolwarm)
CMAP_CW_MASK.set_bad(color='black')

PLT_ERROR_CONFIG = {'cmap': CMAP_CW_MASK}
PLT_DEPTH_CONFIG = {'cmap': CMAP_VIRIDIS_MASK}
PLT_CORR_CONFIG = {'cmap': plt.get_cmap('gist_gray')}

def plot_LLT_map(K1,B1,K2,B2,K3,B3,i,step=0,writer=None):
    fig = plt.figure(figsize=(25, 15))
    K1 = K1.squeeze(0)
    B1 = B1.squeeze(0)
    K2 = K2.squeeze(0)
    B2 = B2.squeeze(0)
    K3 = K3.squeeze(0)
    B3 = B3.squeeze(0)

    ax = plt.subplot(3,2,1)
    ax.imshow(K1 , **PLT_DEPTH_CONFIG)
    ax.set_title('K1')

    ax = plt.subplot(3,2,2)
    ax.imshow(B1 , **PLT_DEPTH_CONFIG)
    ax.set_title('B1')

    ax = plt.subplot(3,2,3)
    ax.imshow(K2 , **PLT_DEPTH_CONFIG)
    ax.set_title('K2')

    ax = plt.subplot(3,2,4)
    ax.imshow(B2 , **PLT_DEPTH_CONFIG)
    ax.set_title('B2')

    ax = plt.subplot(3,2,5)
    ax.imshow(K3 , **PLT_DEPTH_CONFIG)
    ax.set_title('K3')

    ax = plt.subplot(3,2,6)
    ax.imshow(B3 , **PLT_DEPTH_CONFIG)
    ax.set_title('B3')

    fig.tight_layout()
    writer.add_figure(str(i) + '/figure_LLT', fig, step)
    plt.close()



def plot_correlation_warp(img1_gt,warped_img1,img0_1, img2_gt, warped_img2,img0_2, img3_gt,warped_img3,img0_3,
                                            photo_bias1, photo_bias2, photo_bias3, gt_tof_depth,warped_tof_depth_refined, warped_tof_depth, adjusted_img0_1, adjusted_img0_2, adjusted_img0_3, img0_gt,i,step=0,writer=None):
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
    adjusted_img0_1 = adjusted_img0_1.squeeze(0)
    adjusted_img0_2 = adjusted_img0_2.squeeze(0)
    adjusted_img0_3 = adjusted_img0_3.squeeze(0)
    img0_gt = img0_gt.squeeze(0)

    ax = plt.subplot(4,7,1)
    ax.imshow(img0_1 , **PLT_CORR_CONFIG)
    ax.set_title('img0_1')

    ax = plt.subplot(4,7,2)
    ax.imshow(adjusted_img0_1 , **PLT_CORR_CONFIG)
    ax.set_title('adjusted_img0_1')

    ax = plt.subplot(4,7,3)
    ax.imshow(warped_img1 , **PLT_CORR_CONFIG)
    ax.set_title('warped_img1')

    ax = plt.subplot(4,7,4)
    ax.imshow(img1_gt, **PLT_CORR_CONFIG)
    ax.set_title('img1_gt')

    ax = plt.subplot(4,7,5)
    error = warped_img1 - img1_gt
    m = np.max(np.abs(error))
    imsh = ax.imshow(np.abs(error), **PLT_ERROR_CONFIG, vmin=-m, vmax=m)
    ax.set_title('warped_img1 - img1_gt')
    fig.colorbar(imsh, ax=ax, shrink=0.5)

    ax = plt.subplot(4,7,6)
    error = img0_1 - warped_img1
    imsh = ax.imshow(np.abs(error), **PLT_ERROR_CONFIG, vmin=-m, vmax=m)
    ax.set_title('img0_1 - warped_img1')
    fig.colorbar(imsh, ax=ax, shrink=0.5)

    ax = plt.subplot(4,7,7)
    ax.imshow(photo_bias1 * 100, **PLT_CORR_CONFIG)
    ax.set_title('photo_bias1')

    ax = plt.subplot(4,7,8)
    ax.imshow(img0_2 , **PLT_CORR_CONFIG)
    ax.set_title('img0_2')

    ax = plt.subplot(4,7,9)
    ax.imshow(adjusted_img0_2 , **PLT_CORR_CONFIG)
    ax.set_title('adjusted_img0_2')

    ax = plt.subplot(4,7,10)
    ax.imshow(warped_img2 , **PLT_CORR_CONFIG)
    ax.set_title('warped_img2')

    ax = plt.subplot(4,7,11)
    ax.imshow(img2_gt, **PLT_CORR_CONFIG)
    ax.set_title('img2_gt')

    ax = plt.subplot(4,7,12)
    error = warped_img2 - img2_gt
    m = np.max(np.abs(error))
    imsh = ax.imshow(np.abs(error), **PLT_ERROR_CONFIG, vmin=-m, vmax=m)
    ax.set_title('warped_img2 - img2_gt')
    fig.colorbar(imsh, ax=ax, shrink=0.5)

    ax = plt.subplot(4,7,13)
    error = img0_2 - warped_img2
    imsh = ax.imshow(np.abs(error), **PLT_ERROR_CONFIG, vmin=-m, vmax=m)
    ax.set_title('img0_2 - warped_img2')
    fig.colorbar(imsh, ax=ax, shrink=0.5)

    ax = plt.subplot(4,7,14)
    ax.imshow(photo_bias2 * 100, **PLT_CORR_CONFIG)
    ax.set_title('photo_bias2')

    ax = plt.subplot(4,7,15)
    ax.imshow(img0_3 , **PLT_CORR_CONFIG)
    ax.set_title('img0_3')

    ax = plt.subplot(4,7,16)
    ax.imshow(adjusted_img0_3 , **PLT_CORR_CONFIG)
    ax.set_title('adjusted_img0_3')

    ax = plt.subplot(4,7,17)
    ax.imshow(warped_img3 , **PLT_CORR_CONFIG)
    ax.set_title('warped_img3')

    ax = plt.subplot(4,7,18)
    ax.imshow(img3_gt, **PLT_CORR_CONFIG)
    ax.set_title('img3_gt')

    ax = plt.subplot(4,7,19)
    error = warped_img3 - img3_gt
    m = np.max(np.abs(error))
    imsh = ax.imshow(np.abs(error), **PLT_ERROR_CONFIG, vmin=-m, vmax=m)
    ax.set_title('warped_img3 - img3_gt')
    fig.colorbar(imsh, ax=ax, shrink=0.5)

    ax = plt.subplot(4,7,20)
    error = img0_3 - warped_img3
    imsh = ax.imshow(np.abs(error), **PLT_ERROR_CONFIG, vmin=-m, vmax=m)
    ax.set_title('img0_3 - warped_img3')
    fig.colorbar(imsh, ax=ax, shrink=0.5)

    ax = plt.subplot(4,7,21)
    ax.imshow(photo_bias3 * 100, **PLT_CORR_CONFIG)
    ax.set_title('photo_bias3')

    ax = plt.subplot(4,7,22)
    ax.imshow(gt_tof_depth , **PLT_DEPTH_CONFIG)
    ax.set_title('gt_tof_depth')

    ax = plt.subplot(4,7,23)
    ax.imshow(warped_tof_depth , **PLT_DEPTH_CONFIG)
    ax.set_title('warped_tof_depth')

    ax = plt.subplot(4,7,24)
    ax.imshow(warped_tof_depth_refined , **PLT_DEPTH_CONFIG)
    ax.set_title('warped_tof_depth_refined')


    ax = plt.subplot(4,7,25)
    error = warped_tof_depth_refined - gt_tof_depth
    m = 0.4
    imsh = ax.imshow(np.abs(error), **PLT_ERROR_CONFIG, vmin=-m, vmax=m)
    ax.set_title('warped_tof_depth_refined - gt_tof_depth')
    fig.colorbar(imsh, ax=ax, shrink=0.5)

    ax = plt.subplot(4,7,26)
    error = warped_tof_depth - gt_tof_depth
    m = 0.4
    imsh = ax.imshow(np.abs(error), **PLT_ERROR_CONFIG, vmin=-m, vmax=m)
    ax.set_title('warped_tof_depth - gt_tof_depth')
    fig.colorbar(imsh, ax=ax, shrink=0.5)

    ax = plt.subplot(4,7,27)
    ax.imshow(img0_gt, **PLT_CORR_CONFIG)
    ax.set_title('img0_gt')




    fig.tight_layout()
    writer.add_figure(str(i) + '/figure', fig, step)
    plt.close()

def get_learning_rate(step):
    step_threshold = 2000.
    if step < step_threshold:
        mul = step / step_threshold
        return 2e-4 * mul
    else:
        mul = np.cos((step - step_threshold) / (args.epoch * args.step_per_epoch - step_threshold) * math.pi) * 0.5 + 0.5
        return (2e-4 - 2e-6) * mul+ 2e-6 

def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)

def train(model, local_rank):
    if local_rank == 0:
        writer = SummaryWriter('train')
        writer_train_ref = SummaryWriter('train_ref')
        writer_val_ref = SummaryWriter('val_ref')
        writer_val = SummaryWriter('validate')
    else:
        writer = None
        writer_val = None

    if(args.load_flownet):
        model.load_model('train_log', -1)
    step = 0
    nr_eval = 0
    dataset = ToFDataset('train')
    sampler = DistributedSampler(dataset)
    train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True, sampler=sampler, prefetch_factor = 6)
    args.step_per_epoch = train_data.__len__()
    dataset_val = ToFDataset('test')
    val_data = DataLoader(dataset_val, batch_size=args.batch_size, pin_memory=True, num_workers=8)
    print('training...')
    time_stamp = time.time()
    for epoch in range(args.epoch):
        sampler.set_epoch(epoch)
        for i, data in enumerate(train_data):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            data_gpu = data

            data_gpu = data_gpu.to(device, non_blocking=True)


            learning_rate = (get_learning_rate(step) * args.world_size / 6) * (args.batch_size / 8) * 0.8

            warped_img1,warped_img2,warped_img3,info = model.update(data_gpu, learning_rate, training=True)

            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            if local_rank == 0:
                print('epoch:{} {}/{} time:{:.2f}+{:.2f} loss_l1_photo_other_phi_refined:{:.4e} loss_l1_photo_other_phi:{:.4e} depth_loss:{:.4e} LLT loss:{:.4e}'.format(epoch, i, args.step_per_epoch, data_time_interval, train_time_interval,info['loss_l1_photo_other_phi_refined'],  info['loss_l1_photo_other_phi'], info['depth_loss'], info['loss_mi']))
                #evaluate(model, val_data, step, local_rank, writer_val, writer_val_ref)
            step += 1
            
            
        nr_eval += 1

        if nr_eval % 2 == 0 and local_rank == 0:
            val_loss = evaluate(model, val_data, step, local_rank, writer_val, writer_val_ref)
            model.save_model(log_path, local_rank, val_loss, nr_eval)
        dist.barrier()
        torch.cuda.empty_cache()

def evaluate(model, val_data, nr_eval, local_rank, writer_val, writer_val_ref):
    loss_l1_photo_other_phi = []
    loss_l1_photo_other_phi_refined = []
    depth_loss = []
    loss_mi = []

    time_stamp = time.time()
    for i, data in enumerate(val_data):
        data_gpu = data
        data_gpu = data_gpu.to(device, non_blocking=True)    


        with torch.no_grad():
            warped_img1,warped_img2,warped_img3,info = model.update(data_gpu,  training=False)

        loss_l1_photo_other_phi.append(info['loss_l1_photo_other_phi'].cpu().numpy())
        loss_l1_photo_other_phi_refined.append(info['loss_l1_photo_other_phi_refined'].cpu().numpy())
        depth_loss.append(info['depth_loss'].cpu().numpy())
        loss_mi.append(info['loss_mi'].cpu().numpy())
        if (i == 5) and local_rank == 0:
            img0_gt = (data_gpu[:, 4:5].detach().cpu().numpy() )
            img1_gt = (data_gpu[:, 5:6].detach().cpu().numpy() )
            img2_gt = (data_gpu[:, 6:7].detach().cpu().numpy() )
            img3_gt = (data_gpu[:, 7:8].detach().cpu().numpy() )
            img0_1 = (data_gpu[:, 1:2].detach().cpu().numpy() )
            img0_2 = (data_gpu[:, 2:3].detach().cpu().numpy() )
            img0_3 = (data_gpu[:, 3:4].detach().cpu().numpy() )

            flow1 = info['flow1'].permute(0, 2, 3, 1).detach().cpu().numpy()
            flow2 = info['flow2'].permute(0, 2, 3, 1).detach().cpu().numpy()
            flow3 = info['flow3'].permute(0, 2, 3, 1).detach().cpu().numpy()
            photo_bias1 = info['photo_bias1'].detach().cpu().numpy()
            photo_bias2 = info['photo_bias2'].detach().cpu().numpy()
            photo_bias3 = info['photo_bias3'].detach().cpu().numpy()
            warped_img1 = warped_img1.detach().cpu().numpy()
            warped_img2 = warped_img2.detach().cpu().numpy()
            warped_img3 = warped_img3.detach().cpu().numpy()
            gt_tof_depth = info['gt_tof_depth'].detach().cpu().numpy()
            warped_tof_depth = info['warped_tof_depth'].detach().cpu().numpy()
            warped_tof_depth_refined = info['warped_tof_depth_refined'].detach().cpu().numpy()
            adjusted_img0_1 = info['adjusted_img0_1'].detach().cpu().numpy()
            adjusted_img0_2 = info['adjusted_img0_2'].detach().cpu().numpy()
            adjusted_img0_3 = info['adjusted_img0_3'].detach().cpu().numpy()
            K1 = info['K1'].detach().cpu().numpy()
            B1 = info['B1'].detach().cpu().numpy()
            K2 = info['K2'].detach().cpu().numpy()
            B2 = info['B2'].detach().cpu().numpy()
            K3 = info['K3'].detach().cpu().numpy()
            B3 = info['B3'].detach().cpu().numpy()
            for j in range(7):
                plot_correlation_warp(img1_gt[j],warped_img1[j],img0_1[j], img2_gt[j], warped_img2[j],img0_2[j], img3_gt[j],warped_img3[j],img0_3[j],photo_bias1[j], photo_bias2[j], photo_bias3[j],gt_tof_depth[j], warped_tof_depth_refined[j], warped_tof_depth[j],adjusted_img0_1[j], adjusted_img0_2[j], adjusted_img0_3[j],img0_gt[j], j, nr_eval, writer_val)
                plot_LLT_map(K1[j],B1[j],K2[j],B2[j],K3[j],B3[j],j,nr_eval,writer_val)

                writer_val.add_image(str(j) + '/flow1', flow2rgb(flow1[j]), j, dataformats='HWC')
                writer_val.add_image(str(j) + '/flow2', flow2rgb(flow2[j]), j, dataformats='HWC')
                writer_val.add_image(str(j) + '/flow3', flow2rgb(flow3[j]), j, dataformats='HWC')
    

    if local_rank != 0:
        return

    writer_val.add_scalar('loss_l1_photo_other_phi', np.array(loss_l1_photo_other_phi).mean(), nr_eval)
    writer_val.add_scalar('depth loss', np.array(depth_loss).mean(), nr_eval)

    torch.cuda.empty_cache()
    return np.array(loss_l1_photo_other_phi_refined).mean()


        
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=48, type=int)
    parser.add_argument('--batch_size', default=48, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--world_size', default=3, type=int)
    parser.add_argument('--load_flownet', default=False, type=bool, help='load pre-trained flownet')
    args = parser.parse_args()
    torch.distributed.init_process_group(backend="nccl", world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    model = Model(args.local_rank, load_flownet = args.load_flownet)
    train(model, args.local_rank)
