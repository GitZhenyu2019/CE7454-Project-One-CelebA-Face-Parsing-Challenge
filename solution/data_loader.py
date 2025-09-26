# -*- coding: utf-8 -*-
#
# @File:   data_loader.py
# @Author: Zhenyu Yao
# @Date:   2025-09-24
# @Email:  yaozhenyu2019@outlook.com

import os, glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class FaceParsingDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, imsize=512):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))  # return all images ending with ".jpg" and save in deterministic order
        self.mask_dir = mask_dir
        self.has_mask = mask_dir is not None
        self.imsize = imsize

        # transform images
        self.img_tf = T.Compose([
            T.Resize((imsize, imsize), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),                                                   # [0, 1]
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))          # [-1, 1]
        ])

        # transfrom labels
        self.mask_resize = T.Resize((imsize, imsize), interpolation=T.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.has_mask:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(self.mask_dir, stem + ".png")
            mask = Image.open(mask_path).convert("P")
        
        else:
            mask = None
        
        # Resize
        img = self.img_tf(img)
        if mask is not None:
            mask = self.mask_resize(mask)
            mask = torch.from_numpy(np.array(mask, dtype=np.int64))     # [H, W] long tensor
        else:
            mask = torch.tensor(0)      # placeholder

        if mask is not None and idx == 0:
            print("mask unique values:", torch.unique(mask)) # todo modify
        
        return img, mask

def build_loader(img_dir, mask_dir, imsize, batch_size, num_workers, shuffle=True):
    ds = FaceParsingDataset(img_dir=img_dir, mask_dir=mask_dir, imsize=imsize)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)




        

