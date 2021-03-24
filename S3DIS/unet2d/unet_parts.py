""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedConv(nn.Module):
    def __init__(self,in_chl,out_chl,ks,level=3):
        super(DilatedConv, self).__init__()
        self.level = level
        self.conv1 = nn.Sequential(nn.Conv2d(in_chl,out_chl,kernel_size=ks,padding=1),
                                   nn.BatchNorm2d(out_chl),
                                   nn.ReLU(inplace=True)
                                   )
        if self.level==1:
            self.dilat_conv2 = nn.Sequential(nn.Conv2d(out_chl, out_chl, kernel_size=ks, padding=2,dilation=2),
                                             nn.BatchNorm2d(out_chl),
                                             nn.ReLU(inplace=True))
        if self.level==2:
            self.dilat_conv2 = nn.Sequential(nn.Conv2d(out_chl, out_chl, kernel_size=ks, padding=2,dilation=2),
                                             nn.BatchNorm2d(out_chl),
                                             nn.ReLU(inplace=True))
            self.dilat_conv4 = nn.Sequential(nn.Conv2d(out_chl, out_chl, kernel_size=ks, padding=4,dilation=4),
                                             nn.BatchNorm2d(out_chl),
                                             nn.ReLU(inplace=True))
        if self.level==3:
            self.dilat_conv2 = nn.Sequential(nn.Conv2d(out_chl, out_chl, kernel_size=ks, padding=2, dilation=2),
                                             nn.BatchNorm2d(out_chl),
                                             nn.ReLU(inplace=True))
            self.dilat_conv4 = nn.Sequential(nn.Conv2d(out_chl, out_chl, kernel_size=ks, padding=4, dilation=4),
                                             nn.BatchNorm2d(out_chl),
                                             nn.ReLU(inplace=True))
            self.dilat_conv8 = nn.Sequential(nn.Conv2d(out_chl, out_chl, kernel_size=ks, padding=8, dilation=8),
                                             nn.BatchNorm2d(out_chl),
                                             nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(out_chl,out_chl,kernel_size=ks,padding=1),
                                   nn.BatchNorm2d(out_chl),
                                   nn.ReLU(inplace=True)
                                   )
    def forward(self, x):
        x = self.conv1(x)
        if self.level==1:
            x1 = self.dilat_conv2(x)
            out = x1
        if self.level==2:
            x1 = self.dilat_conv2(x)
            x2 = self.dilat_conv4(x1)
            out = x1+x2
        if self.level==3:
            x1 = self.dilat_conv2(x)
            x2 = self.dilat_conv4(x1)
            x3 = self.dilat_conv8(x2)
            out = x1 + x2 + x3
        out = self.conv2(out)
        return out
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=16,numpoints=4096):
#         super(SELayer, self).__init__()
#         self.channel = channel
#         self.numpoints=numpoints
#         # self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel * reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel * reduction, channel, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         m,c = x.shape
#         y = torch.mean(x,dim=0,keepdim=True)
#         y = self.fc(y)
#         y = y.expand(m//self.numpoints,self.numpoints,self.channel).reshape(m,c)
#         return x * y

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel * reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel * reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)