#!/usr/bin/env python3
"""
scripts/train.py

Training script for ImageNet-1K classification on lightweight models
(EfficientNetV2-S, ConvNeXt-Tiny, MobileViT v2 XS, MobileNetV3-Large,
TinyViT-21M, RepVGG-A2) with PyTorch, AMP, mixup, cutmix, RandAugment,
CSV logging, checkpointing, and optional resume/epoch override.
"""
import os, sys
# add project root to PYTHONPATH so that local modules can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import time
import argparse
import yaml
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from models import get_model
from utils.metrics import topk_accuracy
from utils.augmentations import mixup_data, mixup_criterion, cutmix_data


def parse_args():
    parser = argparse.ArgumentParser(description="Train lightweight ImageNet models")
    parser.add_argument("--config", type=str,    required=True, help="Path to YAML config file")
    parser.add_argument("--resume", action="store_true",           help="Resume from latest checkpoint")
    parser.add_argument("--epochs", type=int,    default=None,    help="Override number of epochs")
    parser.add_argument("--device", type=str,    default="cuda:0", help="Device (e.g., 'cuda:0')")
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg):
    # 1. Transforms
    in_sz = cfg["input_size"]
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    train_ops = [transforms.RandomResizedCrop(in_sz), transforms.RandomHorizontalFlip()]
    if cfg.get("randaugment", False):
        train_ops.append(transforms.RandAugment())
    train_ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    train_tf = transforms.Compose(train_ops)

    val_tf = transforms.Compose([
        transforms.Resize(int(in_sz * 256 / 224)),
        transforms.CenterCrop(in_sz),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # 2. Dataset: detect train/val subfolders or fallback to explicit val_dir or split
    root   = cfg["data"]["train_dir"]
    has_t  = os.path.isdir(os.path.join(root, "train"))
    has_v  = os.path.isdir(os.path.join(root, "val"))

    if has_t and has_v:
        tr_root = os.path.join(root, "train")
        vl_root = os.path.join(root, "val")
    else:
        tr_root = root
        vl_root = cfg["data"].get("val_dir", None)

    if vl_root:
        tr_ds = datasets.ImageFolder(tr_root, transform=train_tf)
        vl_ds = datasets.ImageFolder(vl_root, transform=val_tf)
    else:
        # split single-folder
        full_ds = datasets.ImageFolder(tr_root, transform=None)
        n       = len(full_ds)
        val_n   = int(n * cfg["data"].get("val_split", 0.1))
        g       = torch.Generator().manual_seed(cfg["data"].get("seed", 42))
        idx     = torch.randperm(n, generator=g).tolist()
        tr_idx, vl_idx = idx[val_n:], idx[:val_n]

        split_dir = os.path.join(cfg.get("save_dir", "checkpoints"), "split")
        os.makedirs(split_dir, exist_ok=True)
        torch.save(vl_idx, os.path.join(split_dir, "val_idx.pt"))

        tr_ds = Subset(datasets.ImageFolder(tr_root, transform=train_tf), tr_idx)
        vl_ds = Subset(datasets.ImageFolder(tr_root, transform=val_tf),   vl_idx)

    # --- debug print that works for both ImageFolder and Subset ---
    def _get_classes(ds):
        base = ds.dataset if hasattr(ds, "dataset") else ds
        return getattr(base, "classes", [])
    tr_cls = _get_classes(tr_ds)
    vl_cls = _get_classes(vl_ds)
    print(f"--> TRAIN: {len(tr_ds)} samples in {len(tr_cls)} classes")
    print(f"-->   VAL: {len(vl_ds)} samples in {len(vl_cls)} classes")
    # ------------------------------------------------------------

    tr_ld = DataLoader(tr_ds,
                       batch_size=cfg["batch_size"],
                       shuffle=True,
                       num_workers=cfg["workers"],
                       pin_memory=cfg.get("pin_memory", True))
    vl_ld = DataLoader(vl_ds,
                       batch_size=cfg["batch_size"],
                       shuffle=False,
                       num_workers=cfg["workers"],
                       pin_memory=cfg.get("pin_memory", True))

    return tr_ld, vl_ld


def save_checkpoint(state, is_best, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    latest = os.path.join(save_dir, "latest.pth")
    torch.save(state, latest)
    if is_best:
        best = os.path.join(save_dir, "best.pth")
        torch.save(state, best)


def train_one_epoch(model, loader, criterion, optimizer, scaler, cfg, device):
    model.train()
    running_loss = running_corrects = total_samples = 0
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        # Mixup / CutMix
        if cfg.get("mixup", 0) > 0:
            images, ta, tb, lam = mixup_data(images, targets, cfg["mixup"], device)
            use_mix = True
        elif cfg.get("cutmix", 0) > 0:
            images, ta, tb, lam = cutmix_data(images, targets, cfg["cutmix"])
            use_mix = True
        else:
            use_mix = False

        optimizer.zero_grad()
        with autocast(enabled=cfg.get("amp", False)):
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, ta, tb, lam) if use_mix else criterion(outputs, targets)

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        running_loss += loss.item() * bs
        if use_mix:
            acc = lam * topk_accuracy(outputs, ta, topk=(1,))[0] \
                + (1 - lam) * topk_accuracy(outputs, tb, topk=(1,))[0]
            running_corrects += acc * bs / 100.0
        else:
            running_corrects += topk_accuracy(outputs, targets, topk=(1,))[0] * bs / 100.0
        total_samples += bs

    return running_loss / total_samples, running_corrects / total_samples * 100.0


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = running_corrects = total_samples = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss    = criterion(outputs, targets)

            bs = images.size(0)
            running_loss     += loss.item() * bs
            running_corrects += topk_accuracy(outputs, targets, topk=(1,))[0] * bs / 100.0
            total_samples    += bs

    return running_loss / total_samples, running_corrects / total_samples * 100.0


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = get_model(cfg["model"],
                      pretrained=cfg.get("pretrained", False),
                      num_classes=cfg["num_classes"])
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0)).to(device)
    if cfg["optimizer"].lower() == "sgd":
        optimizer = optim.SGD(model.parameters(),
                              lr=cfg["lr"],
                              momentum=cfg.get("momentum", 0.9),
                              weight_decay=cfg.get("weight_decay", 0.0))
    else:
        optimizer = optim.AdamW(model.parameters(),
                                lr=cfg["lr"],
                                weight_decay=cfg.get("weight_decay", 0.0))

    if cfg["scheduler"].lower() == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"], eta_min=float(cfg.get("min_lr", 0)))
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg.get("step_size", 30)),
            gamma=float(cfg.get("gamma", 0.1))
        )

    scaler = GradScaler(enabled=cfg.get("amp", False))

    train_loader, val_loader = build_dataloaders(cfg)
    imgs, labs = next(iter(train_loader))
    print("→ batch image tensor:", imgs.shape)  # expect [B,3,224,224]
    print("→ batch labels:", labs[:10].tolist())  # should be integers 0…999
    print("→ class names (first 10):", train_loader.dataset.classes[:10])
    os.makedirs(os.path.dirname(cfg["log_file"]), exist_ok=True)
    with open(cfg["log_file"], "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    best_acc    = 0.0
    start_epoch = 1

    # optional resume
    if args.resume:
        latest_ckpt = os.path.join(cfg["save_dir"], "latest.pth")
        if os.path.isfile(latest_ckpt):
            print(f"Resuming from {latest_ckpt}")
            ck = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(ck["model_state_dict"])
            optimizer.load_state_dict(ck["optimizer_state_dict"])
            scaler.load_state_dict(ck["scaler_state_dict"])
            start_epoch = ck["epoch"] + 1
            best_acc    = ck.get("best_acc", 0.0)
            print(f"Resumed: starting at epoch {start_epoch}, best acc {best_acc:.2f}%")
        else:
            print(f"[!] --resume passed but no checkpoint found at {latest_ckpt}; starting fresh.")

    # training loop
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        t0 = time.time()
        print(f"[DEBUG] lr = {optimizer.param_groups[0]['lr']}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                optimizer, scaler, cfg, device)
        val_loss,   val_acc   = validate(model,   val_loader,   criterion, device)

        scheduler.step()

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc

        save_checkpoint({
            "epoch": epoch,
            "model_state_dict":    model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict":    scaler.state_dict(),
            "best_acc":             best_acc,
        }, is_best, cfg["save_dir"])

        with open(cfg["log_file"], "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                f"{train_loss:.4f}", f"{train_acc:.2f}",
                f"{val_loss:.4f}",   f"{val_acc:.2f}"
            ])

        print(
            f"Epoch [{epoch}/{cfg['epochs']}]: "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
            f"Time: {time.time() - t0:.1f}s"
        )

    print(f"Training complete. Best Validation Acc: {best_acc:.2f}%")
    print(f"Logs saved to: {cfg['log_file']}")
    print(f"Checkpoints saved to: {cfg['save_dir']}")


if __name__ == "__main__":
    main()
