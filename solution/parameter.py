# -*- coding: utf-8 -*-
#
# @File:   parameter.py
# @Author: Zhenyu Yao
# @Date:   2025-09-24
# @Email:  yaozhenyu2019@outlook.com

import argparse

def str2bool(v):
    return str(v).lower() in ("true", "1", "yes")

def get_parameters():
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument("--train_images", type=str, default="./dev-public/train/images")
    parser.add_argument("--train_masks", type=str, default="./dev-public/train/masks")
    parser.add_argument("--val_images", type=str, default="./dev-public/test/images")
    parser.add_argument("--save_dir", type=str, default="./outputs")
    parser.add_argument("--masks_out_dir", type=str, default="./masks")
    parser.add_argument("--ckpt_path", type=str, default="./solution/ckpt.pth")
    parser.add_argument("--log_dir", type=str, default="./runs")    # TensorBoard

    # Model
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--base_channels", type=int, default=15)    # make sure that #trainable params<1,821,085
    parser.add_argument("--dropout_p", type=float, default=0.05)     # dropout during training for regularisation

    # Train
    parser.add_argument("--epochs", type=int, default=110)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--imsize", type=int, default=512)      # for CelebAMask-HQ dataset

    # Optim
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.95)    # Adam momentum
    parser.add_argument("--beta2", type=float, default=0.999)    # Adam momentum
    parser.add_argument("--weight_decay", type=float, default=1e-4)     # L2 regularisation
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    # Others
    parser.add_argument("--seed", type=int, default=68)
    parser.add_argument("--use_tensorboard", type=str, default="true")
    parser.add_argument("--device", type=str, default="cuda")   # "cuda" or "cpu"

    # Inference (Codabench invoke run.py)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--weights", type=str, default="./solution/ckpt.pth")

    args = parser.parse_args()
    args.use_tensorboard = str2bool(args.use_tensorboard)
    return args

















