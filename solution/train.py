# -*- coding: utf-8 -*-
#
# @File:   train.py
# @Author: Zhenyu Yao
# @Date:   2025-09-24
# @Email:  yaozhenyu2019@outlook.com

from parameter import get_parameters
from trainer import Trainer
import os

if __name__ == "__main__":
    cfg = get_parameters()
    print(cfg)
    print("CWD =", os.getcwd())
    print("log_dir =", cfg.log_dir)
    trainer = Trainer(cfg)
    trainer.train()