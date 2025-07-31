#!/usr/bin/env python3
"""
scripts/eval.py

Evaluation script for trained ImageNet-1K models.
Computes Top-1/Top-5 accuracy and optionally generates a confusion matrix.

Example:
  python -m scripts/eval.py \
    --model efficientnetv2_s \
    --checkpoint checkpoints/efficientnetv2s/best.pth \
    --data_dir data/imagenet1k \
    --confusion_matrix
"""
import os, sys
# add project root to PYTHONPATH so that local modules can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import argparse
import json
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from models import get_model
from utils.metrics import topk_accuracy

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ImageNet model")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g., efficientnetv2_s, convnext_tiny)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to ImageNet-1K folder containing train/ & val/ or just subfolders")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Computation device")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of data loader workers")
    parser.add_argument("--confusion_matrix", action="store_true",
                        help="Whether to compute and save a confusion matrix")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory to save evaluation results")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model = get_model(args.model, pretrained=False, num_classes=1000)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.to(device).eval()

    # Data transforms
    input_size = 224
    val_tf = transforms.Compose([
        transforms.Resize(int(input_size * 256 / 224)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Auto-detect val/ subfolder
    root = args.data_dir
    if os.path.isdir(os.path.join(root, "val")):
        val_root = os.path.join(root, "val")
    else:
        val_root = root

    # DataLoader
    val_dataset = datasets.ImageFolder(val_root, val_tf)
    loader = DataLoader(val_dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True)

    # Run evaluation
    total = 0
    top1_c = 0.0
    top5_c = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            acc1, acc5 = topk_accuracy(outputs, labels, topk=(1,5))
            bs = images.size(0)
            top1_c += acc1 * bs / 100.0
            top5_c += acc5 * bs / 100.0
            total += bs

            if args.confusion_matrix:
                preds = outputs.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

    top1 = top1_c / total * 100.0
    top5 = top5_c / total * 100.0

    # Save results JSON
    result = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "top1": top1,
        "top5": top5,
        "samples": total
    }
    result_path = os.path.join(args.results_dir, f"{args.model}_eval.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Top-1: {top1:.2f}%, Top-5: {top5:.2f}%")
    print(f"Saved evaluation results to {result_path}")

    # Confusion matrix (optional)
    if args.confusion_matrix:
        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        cm = confusion_matrix(labels, preds)

        # Save raw confusion matrix
        cm_path = os.path.join(args.results_dir, f"{args.model}_confusion_matrix.npy")
        np.save(cm_path, cm)

        # Normalize so each row sums to 1 (proportion per true class)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        # Subsample ticks (every N classes)
        n = cm.shape[0]
        tick_step = max(1, n // 20)  # ~20 ticks
        ticks = list(range(0, n, tick_step))
        tick_labels = [str(i) for i in ticks]

        # Plot normalized heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            cm_norm,
            ax=ax,
            cmap="Blues",
            vmin=0.0,
            vmax=cm_norm.max(),
            cbar=True,
            cbar_kws={"label": "Proportion"}
        )
        ax.set_title(f"Normalized Confusion Matrix: {args.model}")
        ax.set_xlabel("Predicted Class Index")
        ax.set_ylabel("True Class Index")
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=8)
        ax.set_yticks(ticks)
        ax.set_yticklabels(tick_labels, rotation=0, fontsize=8)

        # Save figure
        fig_path = os.path.join("plots", f"{args.model}_confusion_matrix.png")
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Confusion matrix (raw) saved to {cm_path}")
        print(f"Normalized confusion matrix plot saved to {fig_path}")


if __name__ == "__main__":
    main()
