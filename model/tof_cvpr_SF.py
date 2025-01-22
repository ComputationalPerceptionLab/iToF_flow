import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
from model.warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from model.IFnet_tof_cvpr_SF import *
from model.tools_layer import *
from model.warplayer import correlation2depth

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class Model:
    def __init__(self, local_rank=-1, load_flownet = False, arbitrary=False):
        self.load_flownet = load_flownet
        self.flownet = IFNet_tof(load_flownet)
        self.device()
        self.optimG = AdamW(filter(lambda p: p.requires_grad, self.flownet.parameters()), lr=1e-6, weight_decay=1e-3)
        self.pre_loss = []
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)
        
        total = sum([param.nelement() for param in self.flownet.parameters()])
        print("Number of parameter: %.2fM" % (total/1e6))

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def intersect_dicts(self, da, db):
        return {k: v for k, v in da.items() if k in db and v.shape == db[k].shape}

    def load_model(self, path, rank=0):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }
        if rank <= 0:
            state_dict = torch.load('{}/flownet.pkl'.format(path), map_location='cpu')
            model_dict = self.flownet.state_dict()
            state_dict = self.intersect_dicts(state_dict, model_dict)
            print(state_dict.keys())
            if len(state_dict) == 0:
                print('load model failed')
            model_dict.update(state_dict)
            self.flownet.load_state_dict(model_dict)
            print('load model from sucess')
    def load_model_evl(self, path, rank=0):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }
            
        if rank <= 0:
            self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path))))
        
    def save_model(self, path, rank=0, val_loss=0, nr_eval = 0):

        if rank == 0:
            checkpoint = { 'optimizer': self.optimG.state_dict()}
            torch.save(checkpoint, '{}/{}_optimizer.pth'.format(path,nr_eval))
            torch.save(self.flownet.state_dict(),'{}/{}flownet_{:04f}.pkl'.format(path, nr_eval,val_loss))
    
    def update(self, imgs, learning_rate=0, mul=1, training=True, flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        if training:
            self.train()
        else:
            self.eval()
        flow1, photo_bias1, flow2, photo_bias2, flow3, photo_bias3, warped_img1, warped_img2, warped_img3 , warped_img1_refined, warped_img2_refined, warped_img3_refined, mask1, mask2, mask3, info= self.flownet((imgs), scale=[4, 2, 1])


        img1_0 = imgs[:, 4:5]
        img1_1 = imgs[:, 5:6]
        img1_2 = imgs[:, 6:7]
        img1_3 = imgs[:, 7:8]
        freq = imgs[:, 8:9]

        max_value = 1

    
        loss_l1_photo_other_phi = (nn.L1Loss()(warped_img1*max_value * mask1, img1_1*max_value * mask1 ) + \
            nn.L1Loss()(warped_img2*max_value * mask2, img1_2*max_value*mask2 ) + \
            nn.L1Loss()(warped_img3*max_value*mask3 , img1_3*max_value *mask3))/3
        loss_l1_photo_other_phi_refined = (nn.L1Loss()(warped_img1_refined*max_value * mask1, img1_1*max_value * mask1) + \
                                        nn.L1Loss()(warped_img2_refined*max_value* mask2 , img1_2*max_value* mask2 ) + \
                                        nn.L1Loss()(warped_img3_refined*max_value* mask3 , img1_3*max_value * mask3))/3
        

        #深度loss
        gt_tof_depth = correlation2depth(img1_0, img1_1,img1_2,img1_3, freq)
        #warped_tof_depth = correlation2depth(img1_0, warped_img1, warped_img2, warped_img3, freq)
        warped_tof_depth_refined = correlation2depth(img1_0, warped_img1_refined, warped_img2_refined, warped_img3_refined, freq)
        warped_tof_depth = correlation2depth(img1_0, warped_img1, warped_img2, warped_img3, freq)
        mask = (gt_tof_depth != 0) * (gt_tof_depth < 1e2) * (warped_tof_depth_refined != 0) * (warped_tof_depth_refined < 1e2) * mask1 * mask2 * mask3
        depth_loss = nn.L1Loss()(warped_tof_depth_refined*mask, gt_tof_depth*mask)


        loss_total = loss_l1_photo_other_phi_refined + info['loss_mi']+ loss_l1_photo_other_phi 
        
        if training:
            self.optimG.zero_grad()
            loss_G = loss_total
            loss_G.backward()
            self.optimG.step()
        
        return warped_img1_refined, warped_img2_refined, warped_img3_refined, {
                'flow1': flow1,
                'flow2': flow2,
                'flow3': flow3,
                'photo_bias1': photo_bias1,
                'photo_bias2': photo_bias2,
                'photo_bias3': photo_bias3,
                'loss_l1_photo_other_phi': loss_l1_photo_other_phi ,
                'loss_l1_photo_other_phi_refined': loss_l1_photo_other_phi_refined,
                'gt_tof_depth': gt_tof_depth*mask,
                'warped_tof_depth_refined': warped_tof_depth_refined*mask,
                'depth_loss': depth_loss,
                'mask': mask,
                'warped_tof_depth': warped_tof_depth*mask,
                'adjusted_img0_0': info['adjusted_img0_0'],
                'adjusted_img0_1': info['adjusted_img0_1'],
                'adjusted_img0_2': info['adjusted_img0_2'],
                'adjusted_img0_3': info['adjusted_img0_3'],
                'loss_mi': info['loss_mi'] * 100,
                'total_time': info['total_time'],
                'K1': info['K1'],
                'K2': info['K2'],
                'K3': info['K3'],
                'B1': info['B1'],
                'B2': info['B2'],
                'B3': info['B3'],
                'warped_img1': warped_img1,
                'warped_img2': warped_img2,
                'warped_img3': warped_img3,
            }