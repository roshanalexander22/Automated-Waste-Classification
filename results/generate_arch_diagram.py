"""Generate high-resolution system architecture diagram for waste classification."""

import os
import matplotlib.patches as patches
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis("off")

# Colors
c_input = "#3498db"
c_prep = "#9b59b6"
c_model1 = "#e67e22"
c_model2 = "#27ae60"
c_eval = "#16a085"
c_out = "#2c3e50"
c_xai = "#c0392b"


def draw_box(x, y, w, h, title, subtitle, color, text_color="white"):
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.1,rounding_size=0.15",
        facecolor=color,
        edgecolor="#2c3e50",
        linewidth=1.5,
        alpha=0.92,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h * 0.62,
        title,
        color=text_color,
        fontsize=10.5,
        fontweight="bold",
        ha="center",
        va="center",
    )
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.28,
            subtitle,
            color=text_color,
            fontsize=8.5,
            ha="center",
            va="center",
            style="italic",
        )


def draw_arrow(x1, y1, x2, y2, label=""):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(facecolor="#2c3e50", edgecolor="#2c3e50", width=1.8, headwidth=7, headlength=7, shrink=0.05),
    )
    if label:
        ax.text((x1 + x2) / 2, ((y1 + y2) / 2) + 0.15, label, fontsize=8, ha="center", va="bottom", fontweight="bold")


# 1. Input Stage
draw_box(0.5, 4.2, 2.2, 1.8, "Raw Waste Input", "Garbage / User Capture\n(512x384 RGB)", c_input)

# 2. Preprocessing & Augmentation
draw_box(3.2, 4.2, 2.3, 1.8, "Preprocessing Pipeline", "Resize 224x224\nRandCrop / Flip / Jitter\nImageNet Normalization", c_prep)
draw_arrow(2.7, 5.1, 3.2, 5.1)

# 3. Architectures (Dual Pipeline)
draw_box(6.1, 4.8, 2.7, 1.6, "Transfer Learning Model", "MobileNetV2 (Pretrained)\nInverted Residual Blocks\n2.55M Parameters", c_model2)
draw_box(6.1, 2.6, 2.7, 1.6, "Baseline CNN Model", "4-Stage ConvNet (Scratch)\nConv2D + BatchNorm + ReLU\n652K Parameters", c_model1)

draw_arrow(5.5, 5.3, 6.1, 5.6, "Transfer")
draw_arrow(5.5, 4.9, 6.1, 3.4, "Baseline")

# 4. Explainable AI (Grad-CAM)
draw_box(6.1, 0.4, 2.7, 1.6, "Explainable AI (Grad-CAM)", "Feature Activation Hooks\nGradient Backprop\nVisual Attention Map", c_xai)
draw_arrow(7.45, 4.8, 7.45, 2.0, "Gradients")

# 5. Classification Head
draw_box(9.4, 4.2, 2.0, 1.8, "Classification Head", "Adaptive Pooling\nDropout (0.3)\nDense Logits (6 Classes)", c_eval)
draw_arrow(8.8, 5.6, 9.4, 5.3)
draw_arrow(8.8, 3.4, 9.4, 4.9)

# 6. Sustainability Decision Engine
draw_box(11.8, 3.0, 2.0, 3.0, "Sustainability Engine", "Bin Color Mapping\nDisposal Actions\nCO2 Benefit Calculation\nStreamlit UI & API", c_out)
draw_arrow(11.4, 5.1, 11.8, 5.1, "Class")
draw_arrow(8.8, 1.2, 11.8, 3.5, "Heatmap")

# Global Titles
plt.title(
    "Automated Waste Classification System Architecture & Explainability Pipeline",
    fontsize=13,
    fontweight="bold",
    pad=15,
)

plt.tight_layout()
plt.savefig("results/system_architecture.png", bbox_inches="tight")
plt.close()
print("[INFO] System architecture diagram saved to results/system_architecture.png")

