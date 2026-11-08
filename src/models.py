"""Model architectures for automated waste classification.

Includes:
1. BaselineCNN: 4-stage convolutional baseline trained from scratch.
2. MobileNetV2WasteClassifier: Pretrained lightweight MobileNetV2 with custom classification head.
3. EfficientNetWasteClassifier: Pretrained EfficientNet-B0 for scalable high-performance classification.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V2_Weights,
    efficientnet_b0,
    mobilenet_v2,
)


class BaselineCNN(nn.Module):
    """Custom 4-stage Convolutional Neural Network baseline trained from scratch."""

    def __init__(self, num_classes: int = 6, dropout_rate: float = 0.35):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.pool = nn.AdaptiveAvgPool2d((2, 2))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256 * 2 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.7),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

    def get_cam_target_layer(self) -> nn.Module:
        """Returns target convolutional layer for Grad-CAM."""
        return self.conv4[0]


class MobileNetV2WasteClassifier(nn.Module):
    """Transfer learning classifier based on MobileNetV2."""

    def __init__(
        self,
        num_classes: int = 6,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        freeze_features: bool = False,
    ):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v2(weights=weights)

        if freeze_features:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.7),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_cam_target_layer(self) -> nn.Module:
        """Returns target convolutional layer for Grad-CAM."""
        return self.backbone.features[-1]


class EfficientNetWasteClassifier(nn.Module):
    """Transfer learning classifier based on EfficientNet-B0."""

    def __init__(
        self,
        num_classes: int = 6,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        freeze_features: bool = False,
    ):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)

        if freeze_features:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_cam_target_layer(self) -> nn.Module:
        """Returns target convolutional layer for Grad-CAM."""
        return self.backbone.features[-1]


def build_model(
    model_name: str = "mobilenetv2",
    num_classes: int = 6,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Factory function for model creation."""
    model_name = model_name.lower().strip()
    if model_name in ["baseline", "baseline_cnn", "cnn"]:
        return BaselineCNN(num_classes=num_classes)
    elif model_name in ["mobilenetv2", "mobilenet_v2", "mobilenet"]:
        return MobileNetV2WasteClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_features=freeze_backbone,
        )
    elif model_name in ["efficientnet", "efficientnet_b0", "efficientnetb0"]:
        return EfficientNetWasteClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_features=freeze_backbone,
        )
    else:
        raise ValueError(
            f"Unsupported model name '{model_name}'. Choose from: baseline, mobilenetv2, efficientnet"
        )


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Compute parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


if __name__ == "__main__":
    b_model = build_model("baseline")
    m_model = build_model("mobilenetv2", pretrained=False)
    e_model = build_model("efficientnet", pretrained=False)

    print("Baseline CNN params:", count_parameters(b_model))
    print("MobileNetV2 params:", count_parameters(m_model))
    print("EfficientNet-B0 params:", count_parameters(e_model))

    dummy = torch.randn(2, 3, 224, 224)
    print("Baseline output shape:", b_model(dummy).shape)
    print("MobileNetV2 output shape:", m_model(dummy).shape)
    print("EfficientNet output shape:", e_model(dummy).shape)
