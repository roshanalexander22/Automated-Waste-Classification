"""Inference pipeline and sustainability recommendation engine for waste classification."""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.dataset import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD
from src.gradcam import GradCAM
from src.models import build_model

SUSTAINABILITY_GUIDE: Dict[str, Dict[str, Any]] = {
    "cardboard": {
        "material_name": "Cardboard & Corrugated Fiberboard",
        "bin_color": "Blue",
        "bin_label": "Blue Bin (Paper & Cardboard)",
        "icon": "📦",
        "recyclable": True,
        "instructions": (
            "1. Flatten all cardboard boxes to save space in collection bins.\n"
            "2. Remove excessive plastic packaging tape, bubble wrap, and styrofoam inserts.\n"
            "3. Ensure the cardboard is clean and dry. Heavily grease-soaked sections must be torn off and composted."
        ),
        "circular_fact": "Cardboard can be recycled 5 to 7 times before wood pulp fibers become too short.",
        "decomposition_time": "2 months (in aerated soil), but landfill burial locks carbon and creates methane.",
        "environmental_impact": "Recycling 1 ton of cardboard saves 17 trees, 7,000 gallons of water, and 4,000 kWh of energy.",
        "co2_saved_kg_per_kg": 0.90,
    },
    "glass": {
        "material_name": "Glass Containers & Bottles",
        "bin_color": "Green",
        "bin_label": "Green Bin (Glass Recycling)",
        "icon": "🍾",
        "recyclable": True,
        "instructions": (
            "1. Rinse empty bottles and jars with a splash of water to remove residue.\n"
            "2. Metal caps and lids can be separated and recycled with metals.\n"
            "3. Do not include ceramic mugs, Pyrex dishware, mirrors, or window panes in curbside glass bins."
        ),
        "circular_fact": "Glass is 100% infinitely recyclable with zero loss in purity or structural quality.",
        "decomposition_time": "1,000,000+ years in natural environments (practically immortal).",
        "environmental_impact": "Cullet (crushed recycled glass) melts at lower temperatures, cutting furnace energy and emissions by up to 30%.",
        "co2_saved_kg_per_kg": 0.315,
    },
    "metal": {
        "material_name": "Aluminum & Steel Cans / Containers",
        "bin_color": "Yellow",
        "bin_label": "Yellow / Grey Bin (Dry Metals)",
        "icon": "🥫",
        "recyclable": True,
        "instructions": (
            "1. Empty all liquid contents and rinse food tins thoroughly.\n"
            "2. Beverage cans can be gently crushed to maximize sorting bin capacity.\n"
            "3. Clean aluminum baking trays and foil balls (larger than 5cm) are widely accepted."
        ),
        "circular_fact": "Recycling aluminum requires 95% less energy than smelting virgin bauxite ore.",
        "decomposition_time": "80 to 200 years for aluminum; 50 to 100 years for tin/steel cans.",
        "environmental_impact": "An aluminum can can be recycled, remanufactured, and back on a supermarket shelf in just 60 days.",
        "co2_saved_kg_per_kg": 9.13,
    },
    "paper": {
        "material_name": "Paper, Newspaper & Office Documents",
        "bin_color": "Blue",
        "bin_label": "Blue Bin (Paper Stream)",
        "icon": "📄",
        "recyclable": True,
        "instructions": (
            "1. Keep paper dry, clean, and unsoiled.\n"
            "2. Remove plastic spiral binders, plastic sleeves, and heavy glue strips.\n"
            "3. Oily food wrappers, waxed paper, and thermal receipts belong in residual trash or organics."
        ),
        "circular_fact": "Reprocessed paper fibers are converted into daily newsprint, carton board, and tissue rolls.",
        "decomposition_time": "2 to 6 weeks under aerated composting conditions.",
        "environmental_impact": "Recycling paper cuts air pollution from papermaking by 73% compared to virgin wood processing.",
        "co2_saved_kg_per_kg": 1.32,
    },
    "plastic": {
        "material_name": "Plastic Bottles, Containers & Polymers",
        "bin_color": "Yellow",
        "bin_label": "Yellow Bin (Plastics & Packaging)",
        "icon": "🧴",
        "recyclable": True,
        "instructions": (
            "1. Empty all remaining beverage or detergent contents and rinse.\n"
            "2. Keep screw caps securely attached to bottles so they do not get lost in sorting trommels.\n"
            "3. Verify SPI resin codes: PET (#1) and HDPE (#2) are highly sought by recyclers worldwide."
        ),
        "circular_fact": "Recycled PET (rPET) flakes are spun into fleece jackets, polyester fibers, and eco-bottles.",
        "decomposition_time": "450 to 1,000 years, breaking down into persistent, toxic microplastics.",
        "environmental_impact": "Recycling 1 ton of plastic saves approximately 5,774 kWh of electricity and 16.3 barrels of oil.",
        "co2_saved_kg_per_kg": 1.50,
    },
    "trash": {
        "material_name": "Non-Recyclable Residual Waste / Contaminants",
        "bin_color": "Black",
        "bin_label": "Black / Dark Grey Bin (General Landfill)",
        "icon": "🗑️",
        "recyclable": False,
        "instructions": (
            "1. Bag and seal securely to prevent windblown municipal litter.\n"
            "2. Check if specialized drop-off points exist for electronic waste, batteries, or hazardous chemicals.\n"
            "3. Practice waste reduction at source by replacing single-use items with durable, reusable alternatives."
        ),
        "circular_fact": "Residual trash goes to sanitary municipal landfills or modern high-efficiency Waste-to-Energy incinerators.",
        "decomposition_time": "Centuries for synthetic composites, rubber, and treated polymers.",
        "environmental_impact": "Landfilled waste produces greenhouse gases (primarily methane); minimizing trash preserves planetary resources.",
        "co2_saved_kg_per_kg": 0.0,
    },
}


class WasteClassifierPredictor:
    """End-to-end inference handler for waste classification with Grad-CAM and eco-insights."""

    def __init__(
        self,
        model_path: str,
        model_name: str = "mobilenetv2",
        device: Optional[str] = None,
        image_size: int = 224,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_name = model_name
        self.image_size = image_size
        self.model = build_model(model_name=model_name, num_classes=len(CLASS_NAMES), pretrained=False)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        self.target_layer = self.model.get_cam_target_layer()
        self.gradcam = GradCAM(self.model, self.target_layer)

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def predict(
        self,
        image_input: Union[str, Image.Image, np.ndarray],
        compute_gradcam: bool = True,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """Classify waste image and generate comprehensive sustainability advisory."""
        if isinstance(image_input, str):
            orig_pil = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            orig_pil = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            orig_pil = image_input.convert("RGB")
        else:
            raise TypeError("Unsupported image_input type. Must be file path, PIL Image, or numpy array.")

        input_tensor = self.transform(orig_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        sorted_indices = np.argsort(probs)[::-1]
        top_idx = int(sorted_indices[0])
        pred_class = CLASS_NAMES[top_idx]
        confidence = float(probs[top_idx])

        top_predictions = [
            {
                "class_name": CLASS_NAMES[idx],
                "probability": float(probs[idx]),
                "percentage": f"{probs[idx] * 100:.2f}%",
            }
            for idx in sorted_indices[:top_k]
        ]

        cam_overlay = None
        cam_mask = None
        if compute_gradcam:
            try:
                # Re-run input tensor with gradients enabled for Grad-CAM
                grad_tensor = input_tensor.clone().detach().requires_grad_(True)
                cam_mask, _, _ = self.gradcam.generate(grad_tensor, target_class=top_idx)
                cam_overlay = GradCAM.overlay_heatmap(orig_pil, cam_mask, alpha=0.5)
            except Exception as e:
                print(f"[WARN] Grad-CAM generation failed: {e}")

        sustainability_info = SUSTAINABILITY_GUIDE.get(pred_class, {})

        return {
            "predicted_class": pred_class,
            "confidence": confidence,
            "confidence_percentage": f"{confidence * 100:.2f}%",
            "top_predictions": top_predictions,
            "all_probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
            "sustainability": sustainability_info,
            "gradcam_overlay": cam_overlay,
            "gradcam_mask": cam_mask,
            "original_image": orig_pil,
        }
