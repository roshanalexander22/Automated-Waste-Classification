"""Unit tests for model architectures, forward passes, and layer resolution."""

import pytest
import torch
import torch.nn as nn

from src.models import BaselineCNN, MobileNetV2WasteClassifier, build_model, count_parameters


def test_baseline_cnn_forward():
    model = BaselineCNN(num_classes=6)
    model.eval()
    dummy = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (4, 6)
    assert not torch.isnan(out).any()


def test_mobilenetv2_forward():
    model = MobileNetV2WasteClassifier(num_classes=6, pretrained=False)
    model.eval()
    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, 6)
    assert not torch.isnan(out).any()


def test_build_model_factory():
    m1 = build_model("baseline")
    assert isinstance(m1, BaselineCNN)

    m2 = build_model("mobilenetv2", pretrained=False)
    assert isinstance(m2, MobileNetV2WasteClassifier)

    params = count_parameters(m1)
    assert params["total"] > 0
    assert params["trainable"] == params["total"]


def test_cam_target_layers():
    m1 = build_model("baseline")
    t1 = m1.get_cam_target_layer()
    assert isinstance(t1, nn.Module)

    m2 = build_model("mobilenetv2", pretrained=False)
    t2 = m2.get_cam_target_layer()
    assert isinstance(t2, nn.Module)

