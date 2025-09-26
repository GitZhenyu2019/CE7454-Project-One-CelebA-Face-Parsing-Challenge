# -*- coding: utf-8 -*-
#
# @File:   model_utils.py
# @Author: Zhenyu Yao
# @Date:   2025-09-24
# @Email:  yaozhenyu2019@outlook.com

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    """one layer in conv block"""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, dropout_p=0.0):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.do = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x); x = self.bn(x); x = self.act(x); x = self.do(x)
        return x
    
class UNetBlock(nn.Module):
    """two convs (3x3) + BN + ReLU (+Dropout)"""
    def __init__(self, in_ch, out_ch, dropout_p=0.0):
        super(UNetBlock, self).__init__()
        self.c1 = ConvBNReLU(in_ch, out_ch, dropout_p=dropout_p)
        self.c2 = ConvBNReLU(out_ch, out_ch, dropout_p=dropout_p)

    def forward(self, x):
        return self.c2(self.c1(x))
    
class UpBlock(nn.Module):
    """Upsample (transpose conv) + concat skip + UNetBlock"""
    def __init__(self, in_ch, out_ch, dropout_p=0.0):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = UNetBlock(out_ch*2, out_ch, dropout_p=dropout_p)
    
    def forward(self, x, skip): 
        x = self.up(x)
        # padding if shapes in skip connection mismatch (odd sizes)
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.block(x)
