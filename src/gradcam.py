"""Gradient-weighted Class Activation Mapping (Grad-CAM) for waste model interpretability."""

from typing import Optional, Tuple

import matplotlib
import matplotlib.cm as cm
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Computes Grad-CAM heatmaps for convolutional neural networks."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """Generate Grad-CAM heatmap array for a single image tensor of shape [1, 3, H, W]."""
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        if target_class is None:
            target_class = int(torch.argmax(probs, dim=1).item())

        confidence = float(probs[0, target_class].item())

        score = output[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations were not captured. Check target_layer.")

        # Global average pooling over gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [1, 1, H_act, W_act]
        cam = F.relu(cam)

        # Upsample to input tensor resolution
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        cam_np = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np, target_class, confidence

    @staticmethod
    def overlay_heatmap(
        pil_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: str = "jet",
    ) -> Image.Image:
        """Blend 2D heatmap mask [0, 1] with the original PIL image."""
        orig_w, orig_h = pil_image.size
        # Resize heatmap if dimensions differ
        if heatmap.shape != (orig_h, orig_w):
            heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
            heatmap_img = heatmap_img.resize((orig_w, orig_h), resample=Image.Resampling.BILINEAR)
            heatmap = np.array(heatmap_img) / 255.0

        try:
            cmap = matplotlib.colormaps[colormap]
        except (AttributeError, KeyError):
            cmap = cm.get_cmap(colormap)
        colored_heatmap = cmap(heatmap)[:, :, :3]  # drop alpha channel
        colored_heatmap = (colored_heatmap * 255).astype(np.uint8)

        orig_np = np.array(pil_image.convert("RGB"))
        blended = (alpha * colored_heatmap + (1.0 - alpha) * orig_np).astype(np.uint8)
        return Image.fromarray(blended)
