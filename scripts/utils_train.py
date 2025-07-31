import os
import csv
import torch


def save_checkpoint(state, is_best, save_dir):
    """
    Save model checkpoint.
    - latest.pth always updated
    - best.pth updated when is_best=True
    """
    os.makedirs(save_dir, exist_ok=True)
    latest_path = os.path.join(save_dir, "latest.pth")
    torch.save(state, latest_path)
    if is_best:
        best_path = os.path.join(save_dir, "best.pth")
        torch.save(state, best_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, scaler=None, device="cpu"):
    """
    Load model (and optimizer, scaler) state from checkpoint.
    Returns: start_epoch, best_acc
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    start_epoch = checkpoint.get("epoch", 0)
    best_acc = checkpoint.get("best_acc", 0.0)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return start_epoch, best_acc


def init_csv_logger(log_file, header=None):
    """
    Initialize CSV logger by writing header row.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        if header is None:
            header = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
        writer.writerow(header)


def log_epoch_metrics(log_file, epoch, train_loss, train_acc, val_loss, val_acc):
    """
    Append epoch metrics to CSV logger.
    """
    with open(log_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            f"{train_loss:.4f}",
            f"{train_acc:.2f}",
            f"{val_loss:.4f}",
            f"{val_acc:.2f}"
        ])
