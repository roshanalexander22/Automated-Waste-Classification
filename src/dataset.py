"""Dataset management, stratified splitting, and PyTorch DataLoaders."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def find_dataset_dir(explicit_dir: Optional[str] = None) -> str:
    """Find or download the Garbage Classification dataset directory."""
    if explicit_dir and os.path.exists(explicit_dir):
        return explicit_dir

    # Check environment variable
    env_dir = os.environ.get("DATA_DIR")
    if env_dir and os.path.exists(env_dir):
        return env_dir

    # Check local data folder
    local_candidates = [
        os.path.join(os.getcwd(), "data", "Garbage classification", "Garbage classification"),
        os.path.join(os.getcwd(), "data", "Garbage classification"),
        os.path.join(os.getcwd(), "data", "raw"),
    ]
    for cand in local_candidates:
        if os.path.exists(cand) and any(os.path.exists(os.path.join(cand, cls)) for cls in CLASS_NAMES):
            return cand

    # Check standard kagglehub cache locations
    home = Path.home()
    kagglehub_root = home / ".cache" / "kagglehub" / "datasets" / "asdasdasasdas" / "garbage-classification"
    if kagglehub_root.exists():
        for ver_dir in kagglehub_root.glob("versions/*"):
            target = ver_dir / "Garbage classification" / "Garbage classification"
            if target.exists():
                return str(target)
            alt_target = ver_dir / "Garbage classification"
            if alt_target.exists() and any((alt_target / cls).exists() for cls in CLASS_NAMES):
                return str(alt_target)

    # Attempt automatic download via kagglehub
    try:
        import kagglehub

        print("[INFO] Dataset not found locally. Downloading via kagglehub...")
        download_path = kagglehub.dataset_download("asdasdasasdas/garbage-classification")
        target = Path(download_path) / "Garbage classification" / "Garbage classification"
        if target.exists():
            return str(target)
        return download_path
    except Exception as e:
        raise FileNotFoundError(
            f"Could not locate or download the Garbage Classification dataset: {e}\n"
            "Please configure DATA_DIR in .env or download the dataset manually."
        )


def create_or_load_splits(
    data_dir: str,
    splits_csv_path: str = "data/splits.csv",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    force_recreate: bool = False,
) -> pd.DataFrame:
    """Create a stratified train/val/test split or load existing split manifest.

    Guarantees reproducible splits with zero data leakage.
    """
    if os.path.exists(splits_csv_path) and not force_recreate:
        df = pd.read_csv(splits_csv_path)
        # Check that file paths exist, if moved re-map
        if len(df) > 0 and os.path.exists(df.iloc[0]["filepath"]):
            return df

    # Scan directory
    records = []
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.exists(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                records.append(
                    {
                        "filepath": os.path.abspath(os.path.join(cls_dir, fname)),
                        "filename": fname,
                        "class_name": cls,
                        "label": CLASS_TO_IDX[cls],
                    }
                )

    if not records:
        raise ValueError(f"No valid images found in dataset directory: {data_dir}")

    df = pd.DataFrame(records)

    # First split: train + temp (val + test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=df["label"],
        random_state=seed,
    )

    # Second split: val vs test
    val_rel_ratio = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_rel_ratio),
        stratify=temp_df["label"],
        random_state=seed,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(splits_csv_path)), exist_ok=True)
    combined.to_csv(splits_csv_path, index=False)
    print(
        f"[INFO] Stratified splits saved to {splits_csv_path}: "
        f"Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}, Total={len(combined)}"
    )
    return combined


def get_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return data augmentations for training and validation/testing."""
    train_transform = transforms.Compose(
        [
            transforms.Resize((int(image_size * 1.14), int(image_size * 1.14))),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return train_transform, eval_transform


class WasteDataset(Dataset):
    """PyTorch Dataset for waste images."""

    def __init__(self, df: pd.DataFrame, transform: Optional[transforms.Compose] = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["filepath"]
        label = int(row["label"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Failed to read image at {img_path}: {e}")

        if self.transform:
            image = self.transform(image)

        return image, label, row["filename"], row["class_name"]


def get_dataloaders(
    data_dir: Optional[str] = None,
    splits_csv_path: str = "data/splits.csv",
    batch_size: int = 32,
    num_workers: int = 2,
    image_size: int = 224,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame]:
    """Build train, validation, and test DataLoaders."""
    dataset_dir = find_dataset_dir(data_dir)
    splits_df = create_or_load_splits(
        data_dir=dataset_dir,
        splits_csv_path=splits_csv_path,
        seed=seed,
    )

    train_transform, eval_transform = get_transforms(image_size=image_size)

    train_data = WasteDataset(splits_df[splits_df["split"] == "train"], transform=train_transform)
    val_data = WasteDataset(splits_df[splits_df["split"] == "val"], transform=eval_transform)
    test_data = WasteDataset(splits_df[splits_df["split"] == "test"], transform=eval_transform)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, splits_df


def plot_class_distribution(splits_df: pd.DataFrame, save_path: str = "results/class_distribution.png"):
    """Plot and save class distribution across splits."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    palette = {"train": "#2b5c8f", "val": "#e67e22", "test": "#27ae60"}

    pivot = (
        splits_df.groupby(["class_name", "split"])
        .size()
        .unstack(fill_value=0)
        .reindex(CLASS_NAMES)[["train", "val", "test"]]
    )

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    pivot.plot(
        kind="bar",
        stacked=True,
        color=[palette["train"], palette["val"], palette["test"]],
        edgecolor="black",
        linewidth=0.8,
        ax=ax,
    )

    ax.set_title("Garbage Classification Dataset: Stratified Split Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Waste Material Category", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Images", fontsize=11, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(title="Dataset Partition", frameon=True)

    # Annotate total counts on top of bars
    for i, total in enumerate(pivot.sum(axis=1)):
        ax.text(i, total + 12, str(int(total)), ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Class distribution plot saved to {save_path}")


if __name__ == "__main__":
    d_dir = find_dataset_dir()
    df = create_or_load_splits(d_dir)
    plot_class_distribution(df)

