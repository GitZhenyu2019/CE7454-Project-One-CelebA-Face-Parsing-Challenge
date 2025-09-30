# -*- coding: utf-8 -*-
#
# @File:   color.py
# @Author: Zhenyu Yao
# @Date:   2025-09-24
# @Email:  yaozhenyu2019@outlook.com

import os
import glob
import numpy as np
from PIL import Image

# Define the color palette (RGB colors corresponding to indexes 0 to 18)
color_list = [
    [0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0],
    [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204],
    [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0],
    [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204],
    [0, 51, 0], [255, 153, 51], [0, 204, 0]
]

def make_folder(path):
    os.makedirs(path, exist_ok=True)

def label2color(mask, color_list):
    """Convert a single-channel label image (0~18) to a color RGB image"""
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in enumerate(color_list):
        color_img[mask == idx] = color
    return color_img

if __name__ == "__main__":
    folder_base = "./masks"        # input gray scale mask folder
    folder_save = "./masks_color"  # output color mask folder
    make_folder(folder_save)

    # Iterate through all images
    for file in glob.glob(os.path.join(folder_base, "*.png")):
        mask = np.array(Image.open(file))   # single channel [H,W]
        color_mask = label2color(mask, color_list)

        # save colorize result
        filename = os.path.basename(file)
        save_path = os.path.join(folder_save, filename)
        Image.fromarray(color_mask).save(save_path)
        print(f"Saved {save_path}")
