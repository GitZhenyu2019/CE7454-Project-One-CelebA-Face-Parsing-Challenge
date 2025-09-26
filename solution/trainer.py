# -*- coding: utf-8 -*-
#
# @File:   trainer.py
# @Author: Zhenyu Yao
# @Date:   2025-09-24
# @Email:  yaozhenyu2019@outlook.com

import os, time, datetime, math, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from model import UNet
from data_loader import build_loader
from utils import combined_loss, count_trainable_params, WarmupCosine

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        self.device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device=="cuda" else "cpu")
        
        # data
        self.train_loader = build_loader(cfg.train_images, cfg.train_masks, cfg.imsize, cfg.batch_size,
                                         cfg.num_workers, shuffle=True)
        
        # model
        self.model = UNet(cfg.in_channels, cfg.num_classes, cfg.base_channels, cfg.dropout_p).to(self.device)
        # print out the num of params
        n_params = count_trainable_params(self.model)
        print(f"Model Trainable params: {n_params}")
        assert n_params < 1821085, "Parameter limit exceeded! Must be < 1,821,085."

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                                           weight_decay=cfg.weight_decay)
        
        total_steps = cfg.epochs * max(1, len(self.train_loader))
        warmup_steps = cfg.warmup_epochs * max(1, len(self.train_loader))
        self.scheduler = WarmupCosine(self.optimizer, base_lr=cfg.lr, min_lr=cfg.min_lr,
                                       warmup_steps=warmup_steps, total_steps=total_steps)
        
        # logger
        self.use_tb = cfg.use_tensorboard
        os.makedirs(cfg.log_dir, exist_ok=True)
        self.writer = SummaryWriter(cfg.log_dir) if self.use_tb else None

        os.makedirs(os.path.dirname(cfg.ckpt_path), exist_ok=True)

    def train(self):
        self.model.train()      # train mode
        global_step = 0
        start_time = time.time()
        for epoch in range(self.cfg.epochs):
            for imgs, labels in self.train_loader:
                imgs = imgs.to(self.device, non_blocking=True)          # [B, 3, H, W]
                labels = labels.to(self.device, non_blocking=True)      # [B, H, W]

                logits = self.model(imgs)                               # [B, 19, H, W]
                loss, ce, dl = combined_loss(logits=logits, targets=labels, ce_weight=1.0)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                lr = self.scheduler.step()

                # logs
                if self.use_tb:
                    self.writer.add_scalar("loss/total", loss.item(), global_step)
                    self.writer.add_scalar("loss/ce", ce, global_step)
                    self.writer.add_scalar("loss/dice", dl, global_step)
                    self.writer.add_scalar("lr", lr, global_step)
                
                # print logs every certain steps
                if (global_step+1) % 50 == 0:
                    elapsed = str(datetime.timedelta(seconds=int(time.time()-start_time)))
                    print(f"[{elapsed}] epoch {epoch+1}/{self.cfg.epochs} step {global_step+1}: "
                          f"loss={loss.item():.4f} (ce={ce:.4f}, dice={dl:.4f}) lr={lr:.2e}")
                    
                global_step += 1

        # save model checkpoint
        torch.save({"state_dict": self.model.state_dict()}, self.cfg.ckpt_path)
        print(f"Checkpoint saved to {self.cfg.ckpt_path}")
        if self.writer:
            self.writer.close()







        

        
