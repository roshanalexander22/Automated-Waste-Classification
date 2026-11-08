"""Streamlit Interactive Demonstration Application: Automated Waste Classification System."""

import glob
import json
import os
import sys
from typing import Dict, List, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from src.dataset import CLASS_NAMES
from src.predict import SUSTAINABILITY_GUIDE, WasteClassifierPredictor

# Page Configuration
st.set_page_config(
    page_title="AI Waste Classifier & Recycling Advisor",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b5e20;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4b6584;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #27ae60;
        margin-bottom: 15px;
    }
    .bin-card {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        color: #ffffff;
        font-weight: 600;
    }
    .blue-bin { background-color: #1e88e5; }
    .green-bin { background-color: #43a047; }
    .yellow-bin { background-color: #fbc02d; color: #212121; }
    .black-bin { background-color: #424242; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor(model_name: str, checkpoint_path: str):
    """Cached loader for inference model."""
    if not os.path.exists(checkpoint_path):
        return None
    try:
        return WasteClassifierPredictor(model_path=checkpoint_path, model_name=model_name)
    except Exception as e:
        st.error(f"Failed to load model from {checkpoint_path}: {e}")
        return None


def get_available_checkpoints():
    """Discover trained models."""
    checkpoints = {}
    if os.path.exists("models/best_mobilenetv2.pt"):
        checkpoints["MobileNetV2 (Transfer Learning)"] = ("mobilenetv2", "models/best_mobilenetv2.pt")
    if os.path.exists("models/best_baseline.pt"):
        checkpoints["Custom CNN (Baseline)"] = ("baseline", "models/best_baseline.pt")
    return checkpoints


def render_sustainability_card(info: Dict):
    """Render actionable recycling advice card."""
    bin_class = "blue-bin"
    if info.get("bin_color") == "Green":
        bin_class = "green-bin"
    elif info.get("bin_color") == "Yellow":
        bin_class = "yellow-bin"
    elif info.get("bin_color") == "Black":
        bin_class = "black-bin"

    st.markdown(
        f"""
        <div class="bin-card {bin_class}">
            <h3 style="margin:0; font-size:1.3rem;">{info.get('icon', '♻️')} Designated Disposal: {info.get('bin_label', 'Recycle Bin')}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Disposal Instructions:**")
        st.info(info.get("instructions", "Follow local municipal rules."))
        st.markdown(f"**Circular Economy Fact:** {info.get('circular_fact', 'Recycling conserves natural resources.')}")

    with col2:
        st.markdown(f"**Decomposition Period:** `{info.get('decomposition_time', 'Unknown')}`")
        st.markdown(f"**Climate Benefit:** Saves approx `{info.get('co2_saved_kg_per_kg', 0.0)} kg CO₂` equivalent per kg recycled.")
        st.markdown(f"**Environmental Impact:** {info.get('environmental_impact', 'Supports circular economy.')}")


def main():
    st.markdown('<div class="main-header">♻️ Automated Waste Classification & Recycling System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deep Learning-Powered Material Identification, Explainable AI (Grad-CAM), and Circular Economy Advisory</div>', unsafe_allow_html=True)

    # Sidebar controls
    st.sidebar.image("https://img.icons8.com/color/96/recycle-sign.png", width=70)
    st.sidebar.title("System Controls")

    available_ckpts = get_available_checkpoints()
    if not available_ckpts:
        st.sidebar.warning("No trained checkpoints detected in `models/`. Models will appear once training completes.")
        selected_model_label = None
        predictor = None
    else:
        selected_model_label = st.sidebar.selectbox("Active Neural Architecture", list(available_ckpts.keys()))
        m_name, m_path = available_ckpts[selected_model_label]
        predictor = load_predictor(m_name, m_path)

    compute_cam = st.sidebar.checkbox("Compute Grad-CAM Visual Heatmap", value=True)
    cam_alpha = st.sidebar.slider("Grad-CAM Overlay Blend (Alpha)", min_value=0.2, max_value=0.8, value=0.5, step=0.05)
    conf_threshold = st.sidebar.slider("Confidence Warning Threshold (%)", min_value=40, max_value=90, value=60, step=5)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Dataset & Architecture Specs:**
        - **Dataset**: Stanford TrashNet (Garbage Classification)
        - **Classes (6)**: Cardboard, Glass, Metal, Paper, Plastic, Trash
        - **Evaluation**: 15% Held-out Test Set (Stratified)
        - **Explainability**: Gradient-weighted CAM
        """
    )

    # Main Navigation Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Live Classifier", "📂 Batch Test Explorer", "📊 Model Performance & Diagnostics", "📖 Recycling Guidebook"])

    # TAB 1: Live Classifier
    with tab1:
        st.subheader("Interactive Waste Material Recognition")
        st.write("Upload a waste item image, capture with your camera, or select a sample from our test set.")

        input_method = st.radio("Choose Input Source:", ["Sample Images", "Upload Image", "Use Camera"], horizontal=True)
        img_to_classify = None

        if input_method == "Sample Images":
            sample_files = glob.glob("data/sample_images/*/*.*")
            if sample_files:
                sample_options = {
                    f"{os.path.basename(os.path.dirname(f)).capitalize()} — {os.path.basename(f)}": f
                    for f in sample_files
                }
                chosen_sample_label = st.selectbox("Select a Sample Waste Item:", list(sample_options.keys()))
                img_path = sample_options[chosen_sample_label]
                img_to_classify = Image.open(img_path).convert("RGB")
            else:
                st.info("No sample images found in `data/sample_images/`.")

        elif input_method == "Upload Image":
            uploaded_file = st.file_uploader("Upload Waste Photo (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                img_to_classify = Image.open(uploaded_file).convert("RGB")

        elif input_method == "Use Camera":
            camera_img = st.camera_input("Capture Waste Item with Webcam")
            if camera_img is not None:
                img_to_classify = Image.open(camera_img).convert("RGB")

        if img_to_classify is not None:
            col_img, col_pred = st.columns([1, 1.2])

            with col_img:
                st.markdown("#### Input Item")
                st.image(img_to_classify, use_container_width=True)

            with col_pred:
                st.markdown("#### Model Inference")
                if predictor is None:
                    st.error("No active model checkpoint loaded. Train a model first via `python -m src.train`.")
                else:
                    with st.spinner("Analyzing material composition and generating attention map..."):
                        results = predictor.predict(img_to_classify, compute_gradcam=compute_cam)

                    pred_class = results["predicted_class"]
                    confidence = results["confidence"]
                    conf_pct = confidence * 100

                    # Confidence alert
                    status_type = "success" if conf_pct >= conf_threshold else "warning"
                    icon = results["sustainability"].get("icon", "♻️")

                    st.markdown(
                        f"""
                        <div style="background-color: {'#e8f5e9' if status_type == 'success' else '#fff3e0'};
                                    border-left: 6px solid {'#2e7d32' if status_type == 'success' else '#ef6c00'};
                                    padding: 16px; border-radius: 8px; margin-bottom: 12px;">
                            <h2 style="margin: 0; color: #1b5e20;">{icon} Predicted Category: {pred_class.upper()}</h2>
                            <p style="margin: 4px 0 0 0; font-size: 1.1rem; color: #37474f;">
                                Classification Confidence: <b>{conf_pct:.2f}%</b>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.progress(min(1.0, confidence))

                    # Top-3 Probabilities
                    st.markdown("##### Top Predictions Breakdown")
                    prob_df = pd.DataFrame(results["top_predictions"])
                    prob_df.columns = ["Material Class", "Probability", "Confidence"]
                    prob_df["Material Class"] = prob_df["Material Class"].str.capitalize()
                    st.dataframe(prob_df, hide_index=True, use_container_width=True)

            # Grad-CAM and Sustainability Advice
            st.markdown("---")
            col_cam, col_rec = st.columns([1, 1.2])

            with col_cam:
                st.markdown("#### Explainable AI (Grad-CAM Attention)")
                if compute_cam and results.get("gradcam_overlay") is not None:
                    st.image(
                        results["gradcam_overlay"],
                        caption="Grad-CAM Activation: Warm areas (red/yellow) indicate key visual regions guiding the decision.",
                        use_container_width=True,
                    )
                else:
                    st.info("Grad-CAM overlay is disabled or unavailable.")

            with col_rec:
                st.markdown("#### Sustainability & Recycling Action Guide")
                render_sustainability_card(results["sustainability"])

    # TAB 2: Batch Test Explorer
    with tab2:
        st.subheader("Batch Sample Testing")
        st.write("Run the current model across all 18 curated test samples (3 per class) to verify real-world predictions.")

        if st.button("🚀 Run Batch Evaluation on Curated Samples", key="run_batch"):
            if predictor is None:
                st.error("No active model loaded.")
            else:
                sample_files = sorted(glob.glob("data/sample_images/*/*.*"))
                batch_records = []

                progress_bar = st.progress(0.0)
                for idx, f in enumerate(sample_files):
                    true_cls = os.path.basename(os.path.dirname(f))
                    pil_img = Image.open(f).convert("RGB")
                    pred_res = predictor.predict(pil_img, compute_gradcam=False)

                    is_correct = pred_res["predicted_class"] == true_cls
                    batch_records.append(
                        {
                            "Filename": os.path.basename(f),
                            "True Class": true_cls.capitalize(),
                            "Predicted Class": pred_res["predicted_class"].capitalize(),
                            "Confidence": pred_res["confidence_percentage"],
                            "Status": "✅ Correct" if is_correct else "❌ Misclassified",
                            "image": pil_img,
                        }
                    )
                    progress_bar.progress((idx + 1) / len(sample_files))

                batch_df = pd.DataFrame(batch_records)
                acc = (batch_df["Status"] == "✅ Correct").mean() * 100

                st.success(f"Batch Evaluation Finished! Accuracy on Curated Subset: **{acc:.1f}%** ({sum(batch_df['Status'] == '✅ Correct')}/{len(batch_df)})")

                # Display table
                display_df = batch_df[["Filename", "True Class", "Predicted Class", "Confidence", "Status"]].copy()
                st.dataframe(display_df, use_container_width=True, hide_index=True)

    # TAB 3: Model Performance & Diagnostics
    with tab3:
        st.subheader("Rigorous Model Evaluation on 15% Held-Out Test Set")
        st.write("Examine training curves, confusion matrices, comparative benchmark metrics, and error analysis.")

        # Show metrics if available
        col_m1, col_m2 = st.columns(2)
        base_json = "results/metrics_baseline.json"
        mob_json = "results/metrics_mobilenetv2.json"

        with col_m1:
            st.markdown("##### Custom CNN Baseline")
            if os.path.exists(base_json):
                with open(base_json) as f:
                    d = json.load(f)
                st.metric("Test Accuracy", f"{d['accuracy']*100:.2f}%")
                st.metric("Macro F1-Score", f"{d['macro_f1']*100:.2f}%")
            else:
                st.info("Baseline metrics will display once baseline evaluation completes.")

        with col_m2:
            st.markdown("##### MobileNetV2 (Transfer Learning)")
            if os.path.exists(mob_json):
                with open(mob_json) as f:
                    d = json.load(f)
                st.metric("Test Accuracy", f"{d['accuracy']*100:.2f}%")
                st.metric("Macro F1-Score", f"{d['macro_f1']*100:.2f}%")
            else:
                st.info("MobileNetV2 metrics will display once evaluation completes.")

        # Visual Artifacts
        st.markdown("---")
        diag_col1, diag_col2 = st.columns(2)

        with diag_col1:
            st.markdown("##### Class Distribution (Stratified Split)")
            if os.path.exists("results/class_distribution.png"):
                st.image("results/class_distribution.png", use_container_width=True)

            st.markdown("##### Confusion Matrix (MobileNetV2)")
            if os.path.exists("results/confusion_matrix_mobilenetv2.png"):
                st.image("results/confusion_matrix_mobilenetv2.png", use_container_width=True)
            elif os.path.exists("results/confusion_matrix_baseline.png"):
                st.image("results/confusion_matrix_baseline.png", use_container_width=True)

        with diag_col2:
            st.markdown("##### Training & Validation Curves")
            if os.path.exists("results/training_curves_mobilenetv2.png"):
                st.image("results/training_curves_mobilenetv2.png", use_container_width=True)
            elif os.path.exists("results/training_curves_baseline.png"):
                st.image("results/training_curves_baseline.png", use_container_width=True)

            st.markdown("##### Architecture Comparison (Baseline vs Transfer Learning)")
            if os.path.exists("results/model_comparison_metrics.png"):
                st.image("results/model_comparison_metrics.png", use_container_width=True)

        if os.path.exists("results/gradcam_analysis.png"):
            st.markdown("---")
            st.markdown("##### Multi-Class Explainable AI Gallery (Grad-CAM)")
            st.image("results/gradcam_analysis.png", use_container_width=True)

    # TAB 4: Recycling Guidebook
    with tab4:
        st.subheader("Global Circular Economy & Recycling Guidebook")
        st.write("Comprehensive disposal rules, decomposition timelines, and environmental impact for all 6 categories.")

        for cls in CLASS_NAMES:
            info = SUSTAINABILITY_GUIDE[cls]
            with st.expander(f"{info.get('icon', '♻️')} {cls.capitalize()} — {info.get('material_name', '')}"):
                render_sustainability_card(info)


if __name__ == "__main__":
    main()
