# -*- coding: utf-8 -*-
#
# @File:   utils.py
# @Author: Zhenyu Yao
# @Date:   2025-09-24
# @Email:  yaozhenyu2019@outlook.com

import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ============== metrics & losses ===============
def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    # labels: [B, H, W] long
    return F.one_hot(labels.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()    # (B, C, H, W)

def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps=1e-7) -> torch.Tensor:
    """
    logits: [B,C,H,W], raw scores
    targets: [B,H,W] long (0..C-1)
    mean Dice (1 - F1) over classes: Dice=(2*intersection/union)
    loss is average (1-Dice)    
    """
    num_classes = logits.shape[1]
    probs = torch.softmax(logits, dim=1)
    tgt_oh = one_hot(targets, num_classes=num_classes).to(probs.device)

    dims = (0, 2, 3)
    intersection = torch.sum(probs * tgt_oh, dim=dims)
    union = torch.sum(probs, dim=dims) + torch.sum(tgt_oh, dim=dims)
    dice = (2*intersection + eps) / (union + eps)                           # [C]
    return 1.0 - dice.mean()

def combined_loss(logits, targets, ce_weight=0.7, eps=1e-7):
    """
    loss = 0.7*CrossEntropy + 0.3*Dice
    Cross entropy (pixel-level classification accuracy) + Dice (region overlap)
    """
    ce = F.cross_entropy(logits, targets)
    dl = dice_loss(logits, targets, eps=eps)
    return ce_weight * ce + (1.0 - ce_weight) * dl, ce.item(), dl.item()


# ============== LR Scheduler (Warmup + Cosine) ===============
class WarmupCosine:
    def __init__(self, optimizer, base_lr, min_lr, warmup_steps, total_steps):
        """
        Warmup + cosine annealing two-stage strategy
        Args:
        - optimizer: PyTorch optimizer object (with param_groups inside)
        - base_lr: Peak learning rate reached at the end of warmup
        - min_lr: Minimum learning rate at the end of annealing
        - warmup_steps: Number of steps in the warmup stages
        - total_steps: The total number of steps in the entire scheduling process (including warmup + annealing)
        """
        self.opt = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps+1, total_steps)
        self.step_num = 0                                        # step counter during training

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr = self.base_lr * self.step_num / self.warmup_steps
        else:
            t = (self.step_num - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5*(self.base_lr - self.min_lr)*(1 + math.cos(math.pi * t))
        for pg in self.opt.param_groups:
            pg["lr"] = lr
        return lr


# ========== Palette & Saving ==========
def save_mask_png(id_mask: np.ndarray, out_path: str):
    """
    id_mask: [H,W] uint8, value in 0..18
    save as SINGLE-CHANNEL png images with label id as pixels
    """
    img = Image.fromarray(id_mask.astype(np.uint8), mode="L")
    img.save(out_path)   

def logits_to_id_mask(logits: torch.Tensor) -> np.ndarray:
    """
    logits: [1,C,H,W] -> [H,W] uint8
    can only handle one image/ batch=1
    """
    pred = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    return pred 


# =============== Param counter =====================
def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


