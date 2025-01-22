import torch
import torch.nn as nn
import torch.nn.functional as F
from model.warplayer import *
from model.tools_layer import *
import time
def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        nn.PReLU(out_planes)
    )

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )





class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 2+2+2, 4, 2, 1)
        

    def forward(self, x, flow, scale):
        if scale != 1:
            x = F.interpolate(x, scale_factor = 1. / scale, mode="bilinear", align_corners=False)
        if flow != None:
            flow = F.interpolate(flow, scale_factor = 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)        
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        tmp = F.interpolate(tmp, scale_factor = scale * 2, mode="bilinear", align_corners=False)

        flow1 = tmp[:, :2] * scale * 2
        flow2 = tmp[:, 2:4] * scale * 2
        flow3 = tmp[:, 4:6] * scale * 2

        return flow1, flow2, flow3

class LLT_block_PS(nn.Module):
    def __init__(self, in_planes, c=64):
        super(LLT_block_PS, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock1 = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.convblock2 = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )

        self.ps = nn.PixelShuffle(4)
        
    def forward(self, x, scale= 1):
        if scale != 1:
            x = F.interpolate(x, scale_factor = 1. / scale, mode="bilinear", align_corners=False)  
        x = self.conv0(x)
        x = self.convblock1(x) + x 
        x = self.convblock2(x)
        tmp = self.ps(x)

        LLT1 = tmp[:, :2]
        LLT2 = tmp[:, 2:4]
        LLT3 = tmp[:, 4:6]

        return LLT1, LLT2, LLT3
    
class IFNet_tof(nn.Module):
    def __init__(self, load_flownet = False):
        super(IFNet_tof, self).__init__()
        self.block0 = IFBlock(2+1 +3 +3, c=240)
        self.block1 = IFBlock(15+3, c=150)
        self.block2 = IFBlock(15+3, c=90)


        if(load_flownet== True):
            print('freeze the flownet part')
            for param in self.parameters():
                param.requires_grad = False
        self.unet = Unet_3D_pixelshuffe()
        self.LLT = LLT_block_PS(5, c=96)

    def forward(self, x,  T_period = 3e8/(20*1e6), scale=[4,2,1], timestep=0.5):
        img0_0 = x[:, :1]
        img0_1 = x[:, 1:2]
        img0_2 = x[:, 2:3]
        img0_3 = x[:, 3:4]
        img1_0 = x[:, 4:5]
        freq = x[:, 8:9]

        img0_1_0 = x[:, 9:10]
        img0_2_0 = x[:, 10:11]
        img0_3_0 = x[:, 11:12]
        freq = freq / 100
        
        total_time = 0
        LLT2, LLT3, LLT4 = self.LLT(torch.cat((img0_0, img0_1, img0_2, img0_3, img1_0), 1),1)

        #引入LLT修正
        K1 = LLT2[:, :1]
        B1 = LLT2[:, 1:2]
        adjusted_img0_1 = (K1 * img0_1 + B1)
        K2 = LLT3[:, :1]
        B2 = LLT3[:, 1:2]
        adjusted_img0_2 = (K2 * img0_2 + B2)
        K3 = LLT4[:, :1]
        B3 = LLT4[:, 1:2]
        adjusted_img0_3 = (K3 * img0_3 + B3)
        loss_adjusted = (nn.L1Loss()(adjusted_img0_1, img0_1_0) + nn.L1Loss()(adjusted_img0_2, img0_2_0) + nn.L1Loss()(adjusted_img0_3, img0_3_0))/3

        warped_img1 = img1_0
        flow1 = None 
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow1 != None:
                flow_d1, flow_d2, flow_d3= stu[i](torch.cat(((torch.cat((adjusted_img0_1, adjusted_img0_2,adjusted_img0_3,img0_1, img0_2, img0_3,img0_0, img1_0, warped_img1, warped_img2, warped_img3), 1)), freq), 1), torch.cat(( flow1, flow2, flow3),1), scale=scale[i])
                flow1 = flow1 + flow_d1
                flow2 = flow2 + flow_d2
                flow3 = flow3 + flow_d3
            else:
                flow1, flow2, flow3 = stu[i](torch.cat(((torch.cat(( adjusted_img0_1, adjusted_img0_2,adjusted_img0_3,img0_1, img0_2, img0_3,img0_0, img1_0), 1)), freq), 1), None, scale=scale[i])
            
            warped_img1, mask1 = warp(img0_1, flow1[:, 0:2])
            warped_img2, mask2 = warp(img0_2, flow2[:, 0:2])
            warped_img3, mask3= warp(img0_3, flow3[:, 0:2])
            photo_bias1 = warped_img1
            photo_bias2 = warped_img2
            photo_bias3 = warped_img3
            warped_img1_refined = warped_img1
            warped_img2_refined = warped_img2
            warped_img3_refined = warped_img3

        tmp = self.unet(torch.cat((img0_1, img0_2, img0_3,img0_0, img1_0, warped_img1, warped_img2, warped_img3, flow1, flow2, flow3, freq),1))

        res = tmp[:, :3] * 2 - 1
        warped_img1_refined = torch.clamp(warped_img1 + res[:,:1], 0, 1)
        warped_img2_refined = torch.clamp(warped_img2 + res[:,1:2], 0, 1)
        warped_img3_refined = torch.clamp(warped_img3 + res[:,2:3], 0, 1)
        photo_bias1 = res[:,:1]
        photo_bias2 = res[:,1:2]
        photo_bias3 = res[:,2:3]

        return flow1, photo_bias1, flow2, photo_bias2, flow3, photo_bias3, warped_img1, warped_img2, warped_img3, warped_img1_refined, warped_img2_refined, warped_img3_refined, mask1, mask2, mask3, {
            'loss_mi': loss_adjusted,
            'adjusted_img0_0':adjusted_img0_1,
            'adjusted_img0_1':adjusted_img0_1,
            'adjusted_img0_2':adjusted_img0_2,
            'adjusted_img0_3':adjusted_img0_3,
            'K1':K1,
            'B1':B1,
            'K2':K2,
            'B2':B2,
            'K3':K3,
            'B3':B3,
            'total_time':total_time,
        }