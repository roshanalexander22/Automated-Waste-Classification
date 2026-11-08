"""Comprehensive evaluation engine: test metrics, confusion matrix, error analysis, and Grad-CAM interpretability."""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import torch
import torch.nn.functional as F

from src.dataset import CLASS_NAMES, WasteDataset, get_dataloaders, get_transforms
from src.gradcam import GradCAM
from src.models import build_model


def evaluate_model_on_test(
    model_name: str,
    checkpoint_path: str,
    data_dir: Optional[str] = None,
    splits_csv_path: str = "data/splits.csv",
    batch_size: int = 32,
    results_dir: str = "results",
    device_str: Optional[str] = None,
) -> Dict[str, Any]:
    """Run full evaluation on held-out test split."""
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Evaluating {model_name} from '{checkpoint_path}' on device: {device}")

    # Load model
    model = build_model(model_name=model_name, num_classes=len(CLASS_NAMES), pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # Load test split
    _, _, test_loader, splits_df = get_dataloaders(
        data_dir=data_dir,
        splits_csv_path=splits_csv_path,
        batch_size=batch_size,
        num_workers=0,
    )

    y_true: List[int] = []
    y_pred: List[int] = []
    all_probs: List[np.ndarray] = []
    filenames: List[str] = []
    class_names: List[str] = []

    with torch.no_grad():
        for images, labels, fnames, cnames in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.tolist())
            all_probs.extend(probs)
            filenames.extend(fnames)
            class_names.extend(cnames)

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    all_probs_arr = np.array(all_probs)

    # Core metrics
    accuracy = float(np.mean(y_true_arr == y_pred_arr))
    macro_precision = float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))

    report_dict = classification_report(
        y_true_arr,
        y_pred_arr,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    conf_mat = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(len(CLASS_NAMES))))

    # Save metrics JSON
    metrics_summary = {
        "model_name": model_name,
        "test_samples": len(y_true_arr),
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": {
            cls: {
                "precision": report_dict[cls]["precision"],
                "recall": report_dict[cls]["recall"],
                "f1_score": report_dict[cls]["f1-score"],
                "support": report_dict[cls]["support"],
            }
            for cls in CLASS_NAMES
        },
    }

    metrics_json_path = os.path.join(results_dir, f"metrics_{model_name}.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)

    # Plot Confusion Matrix
    cm_path = os.path.join(results_dir, f"confusion_matrix_{model_name}.png")
    plot_confusion_matrix(conf_mat, model_name, cm_path)

    # Error analysis
    error_csv_path, error_plot_path = perform_error_analysis(
        y_true_arr=y_true_arr,
        y_pred_arr=y_pred_arr,
        all_probs_arr=all_probs_arr,
        filenames=filenames,
        splits_df=splits_df,
        model_name=model_name,
        results_dir=results_dir,
    )

    print("\n" + "=" * 65)
    print(f"EVALUATION RESULTS FOR {model_name.upper()} (HELD-OUT TEST SET)")
    print("=" * 65)
    print(f"Overall Accuracy:  {accuracy * 100:.2f}%")
    print(f"Macro Precision:   {macro_precision * 100:.2f}%")
    print(f"Macro Recall:      {macro_recall * 100:.2f}%")
    print(f"Macro F1-Score:    {macro_f1 * 100:.2f}%")
    print(f"Weighted F1-Score: {weighted_f1 * 100:.2f}%")
    print("=" * 65)

    return {
        "metrics": metrics_summary,
        "confusion_matrix": conf_mat.tolist(),
        "confusion_matrix_path": cm_path,
        "metrics_json_path": metrics_json_path,
        "error_csv_path": error_csv_path,
        "error_plot_path": error_plot_path,
    }


def plot_confusion_matrix(conf_mat: np.ndarray, model_name: str, save_path: str):
    """Plot dual raw counts and normalized percentage confusion matrix heatmaps."""
    cm_norm = conf_mat.astype("float") / conf_mat.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Raw counts
    sns.heatmap(
        conf_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar=False,
        ax=ax1,
        linewidths=0.5,
    )
    ax1.set_title(f"{model_name.upper()}: Confusion Matrix (Raw Counts)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax1.set_ylabel("True Label", fontsize=11, fontweight="bold")
    ax1.tick_params(axis="x", rotation=30)

    # Normalized percentages
    sns.heatmap(
        cm_norm * 100,
        annot=True,
        fmt=".1f",
        cmap="Greens",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar=True,
        ax=ax2,
        linewidths=0.5,
    )
    ax2.set_title(f"{model_name.upper()}: Confusion Matrix (% by Class)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax2.set_ylabel("True Label", fontsize=11, fontweight="bold")
    ax2.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix saved to {save_path}")


def perform_error_analysis(
    y_true_arr: np.ndarray,
    y_pred_arr: np.ndarray,
    all_probs_arr: np.ndarray,
    filenames: List[str],
    splits_df: pd.DataFrame,
    model_name: str,
    results_dir: str,
) -> Tuple[str, str]:
    """Perform detailed failure mode breakdown and visualize misclassified examples."""
    test_df = splits_df[splits_df["split"] == "test"].copy().reset_index(drop=True)
    filename_to_path = dict(zip(test_df["filename"], test_df["filepath"]))

    misclassified = []
    for i in range(len(y_true_arr)):
        if y_true_arr[i] != y_pred_arr[i]:
            fname = filenames[i]
            true_cls = CLASS_NAMES[y_true_arr[i]]
            pred_cls = CLASS_NAMES[y_pred_arr[i]]
            conf_pred = float(all_probs_arr[i, y_pred_arr[i]])
            conf_true = float(all_probs_arr[i, y_true_arr[i]])

            misclassified.append(
                {
                    "filename": fname,
                    "filepath": filename_to_path.get(fname, ""),
                    "true_class": true_cls,
                    "predicted_class": pred_cls,
                    "pred_confidence": conf_pred,
                    "true_confidence": conf_true,
                }
            )

    err_df = pd.DataFrame(misclassified)
    error_csv_path = os.path.join(results_dir, f"error_analysis_{model_name}.csv")
    err_df.to_csv(error_csv_path, index=False)
    print(f"[INFO] Error analysis saved to {error_csv_path} ({len(err_df)} misclassifications)")

    error_plot_path = os.path.join(results_dir, f"error_analysis_samples_{model_name}.png")
    if len(err_df) > 0:
        # Sort by predicted confidence descending (highest confidence mistakes)
        err_df_sorted = err_df.sort_values(by="pred_confidence", ascending=False).reset_index(drop=True)
        samples_to_show = min(6, len(err_df_sorted))

        fig, axes = plt.subplots(2, 3, figsize=(13, 8), dpi=300)
        axes = axes.flatten()

        for idx in range(6):
            ax = axes[idx]
            if idx < samples_to_show:
                row = err_df_sorted.iloc[idx]
                fpath = row["filepath"]
                if os.path.exists(fpath):
                    img = Image.open(fpath).convert("RGB")
                    ax.imshow(img)
                else:
                    ax.text(0.5, 0.5, "Image Not Found", ha="center", va="center")

                title_text = (
                    f"True: {row['true_class'].upper()} ({row['true_confidence']*100:.1f}%)\n"
                    f"Pred: {row['predicted_class'].upper()} ({row['pred_confidence']*100:.1f}%)"
                )
                ax.set_title(title_text, fontsize=10, color="crimson", fontweight="bold")
            ax.axis("off")

        plt.suptitle(f"{model_name.upper()} High-Confidence Failure Modes", fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(error_plot_path, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Misclassified sample collage saved to {error_plot_path}")

    return error_csv_path, error_plot_path


def generate_gradcam_gallery(
    model_name: str,
    checkpoint_path: str,
    splits_csv_path: str = "data/splits.csv",
    save_path: str = "results/gradcam_analysis.png",
    device_str: Optional[str] = None,
):
    """Generate a multi-class Grad-CAM interpretability panel using representative test samples."""
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model = build_model(model_name=model_name, num_classes=len(CLASS_NAMES), pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    gradcam = GradCAM(model, model.get_cam_target_layer())
    _, eval_transform = get_transforms(image_size=224)

    splits_df = pd.read_csv(splits_csv_path)
    test_df = splits_df[splits_df["split"] == "test"]

    fig, axes = plt.subplots(len(CLASS_NAMES), 3, figsize=(10, 16), dpi=300)

    for i, cls in enumerate(CLASS_NAMES):
        cls_samples = test_df[test_df["class_name"] == cls]
        if len(cls_samples) == 0:
            continue
        sample_row = cls_samples.iloc[0]
        fpath = sample_row["filepath"]

        pil_img = Image.open(fpath).convert("RGB")
        tensor_img = eval_transform(pil_img).unsqueeze(0).to(device).requires_grad_(True)

        cam_mask, pred_idx, conf = gradcam.generate(tensor_img)
        pred_cls = CLASS_NAMES[pred_idx]
        overlay = GradCAM.overlay_heatmap(pil_img, cam_mask, alpha=0.55)

        # Col 1: Original Image
        axes[i, 0].imshow(pil_img)
        axes[i, 0].set_title(f"Original: {cls.capitalize()}", fontsize=10, fontweight="bold")
        axes[i, 0].axis("off")

        # Col 2: Heatmap Mask
        im2 = axes[i, 1].imshow(cam_mask, cmap="jet")
        axes[i, 1].set_title("Grad-CAM Activation", fontsize=10, fontweight="bold")
        axes[i, 1].axis("off")

        # Col 3: Blended Overlay
        axes[i, 2].imshow(overlay)
        status_color = "darkgreen" if pred_cls == cls else "crimson"
        axes[i, 2].set_title(f"Pred: {pred_cls.capitalize()} ({conf*100:.1f}%)", fontsize=10, fontweight="bold", color=status_color)
        axes[i, 2].axis("off")

    plt.suptitle(
        f"Explainable AI: {model_name.upper()} Grad-CAM Attention Across Waste Classes",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Grad-CAM gallery saved to {save_path}")


def generate_comparison_plot(
    metrics_baseline_path: str,
    metrics_mobilenet_path: str,
    save_path: str = "results/model_comparison_metrics.png",
):
    """Plot comparison bar charts between Baseline CNN and Transfer Learning."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not (os.path.exists(metrics_baseline_path) and os.path.exists(metrics_mobilenet_path)):
        print("[WARN] Both metric files must exist to generate comparison plot.")
        return

    with open(metrics_baseline_path) as f:
        m_base = json.load(f)
    with open(metrics_mobilenet_path) as f:
        m_trans = json.load(f)

    metric_keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    labels = ["Accuracy", "Macro Prec", "Macro Recall", "Macro F1", "Weighted F1"]
    base_vals = [m_base[k] * 100 for k in metric_keys]
    trans_vals = [m_trans[k] * 100 for k in metric_keys]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    bars1 = ax.bar(x - width / 2, base_vals, width, label="Custom CNN Baseline", color="#7f8c8d", edgecolor="black")
    bars2 = ax.bar(x + width / 2, trans_vals, width, label="MobileNetV2 (Transfer Learning)", color="#27ae60", edgecolor="black")

    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Performance Comparison: Custom CNN Baseline vs. MobileNetV2", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)

    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.2, f"{yval:.1f}%", ha="center", va="bottom", fontsize=9)

    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.2, f"{yval:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Model comparison plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Waste Classification Model")
    parser.add_argument("--model", type=str, default="mobilenetv2", choices=["baseline", "mobilenetv2", "efficientnet"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)

    args = parser.parse_args()
    ckpt = args.checkpoint or f"models/best_{args.model}.pt"

    evaluate_model_on_test(
        model_name=args.model,
        checkpoint_path=ckpt,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
