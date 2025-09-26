# -*- coding: utf-8 -*-
#
# @File:   model.py
# @Author: Zhenyu Yao
# @Date:   2025-09-24
# @Email:  yaozhenyu2019@outlook.com

import torch.nn as nn
from model_utils import UNetBlock, UpBlock

class UNet(nn.Module):
    """
    4 layer U-Net, channel: base, 2b, 4b, 8b, bottleneck=16b
    control total number of parameters
    """
    def __init__(self, in_channels=3, num_classes=19, base_channels=16, dropout_p=0.2):
        super(UNet, self).__init__()
        b = base_channels
        self.enc1 = UNetBlock(in_ch=in_channels, out_ch=b, dropout_p=dropout_p) # encoder block 1
        self.enc2 = UNetBlock(in_ch=b, out_ch=b*2, dropout_p=dropout_p)         # encoder block 2
        self.enc3 = UNetBlock(in_ch=b*2, out_ch=b*4, dropout_p=dropout_p)       # encoder block 3
        self.enc4 = UNetBlock(in_ch=b*4, out_ch=b*8, dropout_p=dropout_p)       # encoder block 4

        self.pool = nn.MaxPool2d(2)                                             # max pooling 2x2

        self.center = UNetBlock(in_ch=b*8, out_ch=b*16, dropout_p=dropout_p)    # bottleneck block

        self.up4 = UpBlock(in_ch=b*16, out_ch=b*8, dropout_p=dropout_p)         # decoder block 1
        self.up3 = UpBlock(in_ch=b*8, out_ch=b*4, dropout_p=dropout_p)          # decoder block 2
        self.up2 = UpBlock(in_ch=b*4, out_ch=b*2, dropout_p=dropout_p)          # decoder block 3
        self.up1 = UpBlock(in_ch=b*2, out_ch=b, dropout_p=dropout_p)            # decoder block 4

        self.final = nn.Conv2d(b, num_classes, kernel_size=1)                   # 1x1 convolution to fit classes

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        c = self.center(self.pool(e4))
        d4 = self.up4(c, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        out = self.final(d1)

        return out


