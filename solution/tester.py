# -*- coding: utf-8 -*-
#
# @File:   tester.py
# @Author: Zhenyu Yao
# @Date:   2025-09-24
# @Email:  yaozhenyu2019@outlook.com

import os, glob
import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image
from model import UNet 
from data_loader import build_loader, FaceParsingDataset
from utils import logits_to_id_mask, save_mask_png, count_trainable_params

class Tester:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device=="cuda" else "cpu")

        # model
        self.model = UNet(cfg.in_channels, cfg.num_classes, cfg.base_channels, cfg.dropout_p).to(self.device)
        n_params = count_trainable_params(self.model)
        print(f"Model Trainable params: {n_params}")
        assert n_params < 1821085, "Parameter limit exceeded! Must be < 1,821,085."

        # load weights from checkpoint
        ckpt = torch.load(cfg.ckpt_path, map_location=self.device)
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        self.model.load_state_dict(state, strict=True)
        
        # make a folder to save prediction masks
        os.makedirs(cfg.masks_out_dir, exist_ok=True)

        # test mode
        self.model.eval()

    @torch.no_grad()        # close gradient record
    def predict_dir(self, img_dir, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        for p in img_paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            mask_path = os.path.join(out_dir, f"{stem}.png")
            self.predict_one(p, mask_path)

    @torch.no_grad()
    def predict_one(self, img_path, out_path):
        # predict a single image
        img = Image.open(img_path).convert("RGB").resize((self.cfg.imsize, self.cfg.imsize), Image.BILINEAR)
        x = (torch.from_numpy(np.array(img)).permute(2,0,1).float()/255.0 - 0.5)/0.5
        x = x.unsqueeze(0).to(self.device)
        logits = self.model(x)
        id_mask = logits_to_id_mask(logits)         # [H,W] uint8
        save_mask_png(id_mask=id_mask, out_path=out_path)
