
import torch
import torch.nn as nn
import torch.nn.functional as F

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



def conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=0, dilation=1):
    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
        )

def deconv3d(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose3d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.PReLU(out_planes)
        )




class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    


def conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=0, dilation=1):
    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
        )

def deconv3d(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose3d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.PReLU(out_planes)
        )
            
class Conv3d(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv3d, self).__init__()
        self.conv1 = conv3d(in_planes, int(out_planes/2), 3, stride, 1)
        self.conv2 = conv3d(int(out_planes/2), out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x





c = 8
class Unet_3D_pixelshuffe(nn.Module):
    def __init__(self):
        super(Unet_3D_pixelshuffe, self).__init__()
        self.c = 8
        self.down0 = Conv3d(1, 2*self.c)
        self.down1 = Conv3d(2*self.c, 4*self.c)
        self.down2 = Conv3d(4*self.c, 8*self.c)

        self.up1 = deconv3d(8*self.c, 4*self.c, [2,4,4],[1,2,2])
        self.up2 = deconv3d(4*self.c, 2*self.c, [2,4,4],[1,2,2])
        self.up3 = deconv3d(2*self.c, self.c, [3,3,3],[1,1,1])
        self.conv = nn.Conv3d(self.c, 1, [3,3,3], [1,1,1], 1)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x):
        x = x.unsqueeze(1)
        s0 = self.down0(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)

        x = self.up1( s2) 
        x = self.up2(torch.cat((x, s1), 2)) 
        x = self.up3(torch.cat((x, s0), 2)) 
        x = self.conv(x)

        x = x.squeeze(1)
        x = self.ps(x)

        return  torch.sigmoid(x)
    
