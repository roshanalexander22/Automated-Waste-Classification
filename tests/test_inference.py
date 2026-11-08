"""Unit tests for sustainability advisory guide and Grad-CAM interpretability module."""

import numpy as np
import pytest
import torch
from PIL import Image

from src.dataset import CLASS_NAMES
from src.gradcam import GradCAM
from src.models import build_model
from src.predict import SUSTAINABILITY_GUIDE


def test_sustainability_guide_coverage():
    for cls in CLASS_NAMES:
        assert cls in SUSTAINABILITY_GUIDE, f"Missing sustainability guidelines for class '{cls}'"
        entry = SUSTAINABILITY_GUIDE[cls]
        assert "bin_color" in entry
        assert "bin_label" in entry
        assert "instructions" in entry
        assert "decomposition_time" in entry
        assert "co2_saved_kg_per_kg" in entry
        assert isinstance(entry["recyclable"], bool)


def test_gradcam_computation():
    model = build_model("baseline")
    target_layer = model.get_cam_target_layer()
    gradcam = GradCAM(model, target_layer)

    dummy_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)
    mask, pred_cls, conf = gradcam.generate(dummy_tensor, target_class=1)

    assert mask.shape == (224, 224)
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0
    assert 0 <= pred_cls < len(CLASS_NAMES)
    assert 0.0 <= conf <= 1.0


def test_gradcam_overlay():
    dummy_img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    dummy_mask = np.ones((100, 100), dtype=np.float32) * 0.5
    overlay = GradCAM.overlay_heatmap(dummy_img, dummy_mask)

    assert overlay.size == (100, 100)
    assert isinstance(overlay, Image.Image)
