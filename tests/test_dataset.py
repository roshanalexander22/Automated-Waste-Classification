"""Unit tests for dataset loading, stratified splitting, and transforms."""

import os
import pandas as pd
import pytest
import torch
from PIL import Image

from src.dataset import CLASS_NAMES, CLASS_TO_IDX, WasteDataset, get_transforms


def test_class_names_and_indices():
    assert len(CLASS_NAMES) == 6
    assert CLASS_NAMES == ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    for idx, name in enumerate(CLASS_NAMES):
        assert CLASS_TO_IDX[name] == idx


def test_splits_no_leakage():
    splits_csv = "data/splits.csv"
    assert os.path.exists(splits_csv), "Splits manifest data/splits.csv must exist."

    df = pd.read_csv(splits_csv)
    assert len(df) == 2527

    train_paths = set(df[df["split"] == "train"]["filepath"])
    val_paths = set(df[df["split"] == "val"]["filepath"])
    test_paths = set(df[df["split"] == "test"]["filepath"])

    # Strict zero data leakage verification
    assert len(train_paths.intersection(val_paths)) == 0, "Data leakage between train and val!"
    assert len(train_paths.intersection(test_paths)) == 0, "Data leakage between train and test!"
    assert len(val_paths.intersection(test_paths)) == 0, "Data leakage between val and test!"


def test_transforms_shapes():
    train_transform, eval_transform = get_transforms(image_size=224)
    dummy_img = Image.new("RGB", (300, 300), color=(128, 128, 128))

    tensor_train = train_transform(dummy_img)
    tensor_eval = eval_transform(dummy_img)

    assert tensor_train.shape == (3, 224, 224)
    assert tensor_eval.shape == (3, 224, 224)
    assert isinstance(tensor_train, torch.Tensor)
    assert isinstance(tensor_eval, torch.Tensor)
