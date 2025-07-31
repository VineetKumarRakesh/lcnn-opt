#!/usr/bin/env python3
"""
scripts/split_train_val.py

Split off a validation set from an existing ImageNet-style train directory
by moving (not copying) a fraction of images per class into a new val folder.

Example:
  python scripts/split_train_val.py \
    --train_dir data/imagenet1k/train \
    --val_dir   data/imagenet1k/val \
    --val_split 0.10 \
    --seed      42
"""

import os
import argparse
import random
import shutil

def parse_args():
    p = argparse.ArgumentParser(
        description="Move a fraction of images from train_dir to val_dir"
    )
    p.add_argument(
        "--train_dir", required=True,
        help="Root train directory with one subfolder per class"
    )
    p.add_argument(
        "--val_dir", required=True,
        help="Destination validation directory (will mirror class subfolders)"
    )
    p.add_argument(
        "--val_split", type=float, default=0.1,
        help="Fraction of each class to move (e.g. 0.1 for 10%)"
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible splits"
    )
    return p.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)

    # Make sure destination exists
    os.makedirs(args.val_dir, exist_ok=True)

    # Iterate over each class folder in train_dir
    for cls in sorted(os.listdir(args.train_dir)):
        src_cls_dir = os.path.join(args.train_dir, cls)
        if not os.path.isdir(src_cls_dir):
            continue

        dst_cls_dir = os.path.join(args.val_dir, cls)
        os.makedirs(dst_cls_dir, exist_ok=True)

        # List all files in this class
        all_imgs = [
            f for f in os.listdir(src_cls_dir)
            if os.path.isfile(os.path.join(src_cls_dir, f))
        ]
        random.shuffle(all_imgs)

        # Compute how many to move
        n_move = int(len(all_imgs) * args.val_split)
        to_move = all_imgs[:n_move]

        # Move them
        for fname in to_move:
            src_path = os.path.join(src_cls_dir, fname)
            dst_path = os.path.join(dst_cls_dir, fname)
            shutil.move(src_path, dst_path)

        print(f"[{cls}] moved {n_move}/{len(all_imgs)} images → {dst_cls_dir}")

if __name__ == "__main__":
    main()
