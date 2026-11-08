"""Training engine for waste classification models with checkpointing and metric logging."""

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dataset import CLASS_NAMES, get_dataloaders
from src.models import build_model, count_parameters


def set_seed(seed: int = 42):
    """Set random seed across libraries for absolute reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> Tuple[float, float]:
    """Execute one training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_cuda = device.type == "cuda"

    for images, labels, _, _ in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on validation split."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    use_cuda = device.type == "cuda"

    with torch.no_grad():
        for images, labels, _, _ in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def plot_and_save_training_curves(history: Dict[str, List[float]], model_name: str, save_path: str):
    """Plot and save loss and accuracy progression curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Loss curve
    ax1.plot(epochs, history["train_loss"], "o-", color="#2980b9", label="Training Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "s--", color="#e74c3c", label="Validation Loss", linewidth=2)
    ax1.set_title(f"{model_name.upper()} Loss Convergence", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Accuracy curve
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", color="#27ae60", label="Train Accuracy", linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "s--", color="#8e44ad", label="Val Accuracy", linewidth=2)
    ax2.set_title(f"{model_name.upper()} Accuracy Progression", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Training curves saved to {save_path}")


def train_model(
    model_name: str = "mobilenetv2",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 5,
    data_dir: str = None,
    splits_csv_path: str = "data/splits.csv",
    save_dir: str = "models",
    results_dir: str = "results",
    seed: int = 42,
) -> Dict[str, Any]:
    """Train waste classifier with full checkpointing and metric logging."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Initiating training for '{model_name}' on device: {device}")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    train_loader, val_loader, _, _ = get_dataloaders(
        data_dir=data_dir,
        splits_csv_path=splits_csv_path,
        batch_size=batch_size,
        num_workers=2 if os.name != "nt" else 0,  # avoid Windows multiprocessing spawn overhead
        seed=seed,
    )

    is_pretrained = model_name.lower() in ["mobilenetv2", "efficientnet"]
    model = build_model(model_name=model_name, num_classes=len(CLASS_NAMES), pretrained=is_pretrained)
    model.to(device)

    params_info = count_parameters(model)
    print(f"[INFO] Model parameters: {params_info['total']:,} total ({params_info['trainable']:,} trainable)")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda")) if device.type == "cuda" else torch.amp.GradScaler("cpu", enabled=False)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rates": [],
        "epoch_times": [],
    }

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    best_checkpoint_path = os.path.join(save_dir, f"best_{model_name}.pt")
    latest_checkpoint_path = os.path.join(save_dir, f"latest_{model_name}.pt")

    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        epoch_dur = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["learning_rates"].append(current_lr)
        history["epoch_times"].append(epoch_dur)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}% | "
            f"LR: {current_lr:.6f} | Time: {epoch_dur:.1f}s"
        )

        # Track best model by validation accuracy (and tie-break by validation loss)
        is_best = val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss)
        if is_best:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "class_names": CLASS_NAMES,
                },
                best_checkpoint_path,
            )
            print(f"  --> Saved new best checkpoint: Val Acc = {val_acc * 100:.2f}% (Epoch {epoch})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[INFO] Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch}.")
                break

    total_training_time = time.time() - start_total_time
    print(
        f"[INFO] Training finished in {total_training_time / 60:.2f} mins. "
        f"Best Val Acc: {best_val_acc * 100:.2f}% at epoch {best_epoch}."
    )

    # Save latest checkpoint
    torch.save(
        {
            "epoch": epoch,
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "val_acc": val_acc,
            "val_loss": val_loss,
            "class_names": CLASS_NAMES,
        },
        latest_checkpoint_path,
    )

    # Save history JSON
    history_path = os.path.join(results_dir, f"history_{model_name}.json")
    with open(history_path, "w") as f:
        json.dump(
            {
                "model_name": model_name,
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "best_val_loss": best_val_loss,
                "total_training_time_seconds": total_training_time,
                "history": history,
            },
            f,
            indent=2,
        )

    # Plot training curves
    curves_path = os.path.join(results_dir, f"training_curves_{model_name}.png")
    plot_and_save_training_curves(history, model_name, curves_path)

    return {
        "model_name": model_name,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "best_checkpoint_path": best_checkpoint_path,
        "history_path": history_path,
        "curves_path": curves_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Waste Classification Model")
    parser.add_argument("--model", type=str, default="mobilenetv2", choices=["baseline", "mobilenetv2", "efficientnet"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default=None)

    args = parser.parse_args()
    train_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        data_dir=args.data_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
