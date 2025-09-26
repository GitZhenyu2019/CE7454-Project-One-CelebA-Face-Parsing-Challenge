# -*- coding: utf-8 -*-
#
# @File:   test.py
# @Author: Zhenyu Yao
# @Date:   2025-09-24
# @Email:  yaozhenyu2019@outlook.com

from parameter import get_parameters
from tester import Tester
import os

if __name__ == "__main__":
    cfg = get_parameters()
    print(cfg)

    tester = Tester(cfg)

    # Inference on the validation/test set locally and save to masks
    tester.predict_dir(cfg.val_images, cfg.masks_out_dir)

    print(f"Done. Predicted masks saved to: {cfg.masks_out_dir}")