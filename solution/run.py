# -*- coding: utf-8 -*-
#
# @File:   run.py
# @Author: Zhenyu Yao
# @Date:   2025-09-24
# @Email:  yaozhenyu2019@outlook.com

# This is for Codabench to invoke from an external program：
# pip install -r solution/requirements.txt
# python3 solution/run.py --input /path/to/input-image.jpg --output /path/to/output-mask.png --weights solution/ckpt.pth

import argparse, os, numpy as np, torch
from PIL import Image
from model import UNet
from utils import logits_to_id_mask, save_mask_png, count_trainable_params

def load_model(weights_path, device, in_channel=3, num_classes=19, base_channels=15, dropout_p=0.05):
    model = UNet(in_channel, num_classes, base_channels, dropout_p).to(device)
    n_params = count_trainable_params(model)
    print(f"Model Trainable params: {n_params}")
    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

@torch.no_grad()
def infer_single(model, img_path, out_path, device, imsize=512):
    # predict a single image
    img = Image.open(img_path).convert("RGB").resize((imsize, imsize), Image.BILINEAR)
    x = (torch.from_numpy(np.array(img)).permute(2,0,1).float()/255.0 - 0.5)/0.5
    x = x.unsqueeze(0).to(device)
    logits = model(x)
    id_mask = logits_to_id_mask(logits)         # [H,W] uint8
    save_mask_png(id_mask, out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--weights", type=str, default="ckpt.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device)
    infer_single(model, args.input, args.output, device)



