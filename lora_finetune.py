#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fine_tune.py

在 Linux 服务器（SSH）上一键微调 SAM-2.1。
放在 sam2/ 根目录下，数据放在 ../dataset/dataset_demo/ 里。

示例：
    cd sam2
    python fine_tune.py \
      --data-dir ../dataset/dataset_demo \
      --use-cluster 0 \
      --num-gpus 1
"""

import os
import argparse
import subprocess
import shutil

def install_requirements():
    subprocess.run(["pip", "install", "roboflow"], check=True)
    subprocess.run(["pip", "install", "-e", ".[dev]"], cwd=".", check=True)
    subprocess.run(["pip", "install", "supervision"], check=True)

def prepare_dataset(src_dir):
    """
    将用户提供的数据集拷贝到 data/train/images 和 data/train/masks
    """
    src_img_dir  = os.path.join(src_dir, "train", "ccm")
    src_mask_dir = os.path.join(src_dir, "train", "instance_mask")
    dst_img_dir  = os.path.join("data", "train", "images")
    dst_mask_dir = os.path.join("data", "train", "masks")

    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_mask_dir, exist_ok=True)

    for fn in os.listdir(src_img_dir):
        if not fn.lower().endswith(".png"):
            continue
        img_path  = os.path.join(src_img_dir, fn)
        mask_path = os.path.join(src_mask_dir, fn)
        if not os.path.isfile(mask_path):
            print(f"⚠️ 找不到对应掩码: {mask_path}，已跳过")
            continue
        shutil.copy(img_path,  os.path.join(dst_img_dir, fn))
        shutil.copy(mask_path, os.path.join(dst_mask_dir, fn))

    print(f"✅ 数据准备完成，训练图像共 {len(os.listdir(dst_img_dir))} 张，掩码共 {len(os.listdir(dst_mask_dir))} 张")

def download_checkpoints():
    ckpt_dir = os.path.join(os.getcwd(), "checkpoints")
    script   = os.path.join(ckpt_dir, "download_ckpts.sh")
    if os.path.isfile(script):
        subprocess.run([script], cwd=ckpt_dir, check=True)
        print("✅ 预训练权重下载完成")
    else:
        print("⚠️ 未找到 download_ckpts.sh，跳过预训练权重下载")

def train(use_cluster, num_gpus):
    cmd = [
        "python", "training/train.py",
        "--config-path", "../configs",    # 指定本地 configs 目录
        "--config-name", "train.yaml",    # 指定文件名
        "--use-cluster", str(use_cluster),
        "--num-gpus",   str(num_gpus),
    ]
    subprocess.run(cmd, check=True)


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune SAM-2.1 on local dataset")
    p.add_argument("--data-dir",    type=str, default="../dataset/dataset_demo",
                     help="本地数据集根目录，内部应含 train/ccm 和 train/instance_mask")
    p.add_argument("--use-cluster", type=int, default=0, help="是否使用集群 (0 or 1)")
    p.add_argument("--num-gpus",    type=int, default=1, help="训练使用的 GPU 数量")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # 1. 安装依赖
    install_requirements()

    # 2. 准备数据
    prepare_dataset(args.data_dir)

    # 3. 下载预训练权重
    download_checkpoints()

    # 4. 启动训练
    print("🚀 开始训练 SAM-2.1 …")
    train(use_cluster=args.use_cluster, num_gpus=args.num_gpus)
    print("🎉 微调完成！请查看 sam2_logs/ 目录中的日志和模型检查点。")
