# Garbage Classification Dataset

## 1. Overview
The **Garbage Classification** dataset (commonly known as **TrashNet**) is a benchmark computer vision dataset created by Gary Thung and Mindy Yang at Stanford University. It is curated to train deep learning models for automated municipal and domestic waste sorting to support circular economy initiatives and reduce landfill contamination.

- **Kaggle Source**: [Garbage Classification on Kaggle](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)
- **Original Source**: Gary Thung, Mindy Yang, *Classification of Trash for Recyclability Status*, CS229 Project Report, Stanford University (2016).
- **License**: Open Source / Educational Research License.

---

## 2. Class Distribution and Statistics
The dataset consists of **2,527 RGB images** distributed across **6 material classes**:

| Class | Count | Percentage | Primary Material Examples | Recyclability Status | Bin Recommendation |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **Cardboard** | 403 | 15.95% | Shipping boxes, cereal boxes, corrugated sheets | Recyclable | Blue Bin (Paper/Cardboard) |
| **Glass** | 501 | 19.83% | Beverage bottles, jars, broken glassware | Recyclable | Green Bin (Glass) |
| **Metal** | 410 | 16.22% | Soda cans, food tins, aerosol containers, aluminum foil | Recyclable | Yellow Bin (Dry Recyclables) |
| **Paper** | 594 | 23.51% | Office documents, newspaper, receipts, magazines | Recyclable (if clean) | Blue Bin (Paper) |
| **Plastic** | 482 | 19.07% | Water bottles, food containers, detergent jugs | Recyclable | Yellow Bin (Plastic/Metal) |
| **Trash** | 137 | 5.42% | Contaminated wrappers, dirty diapers, broken non-recyclables | Landfill / Incineration | Black Bin (General Residual) |
| **Total** | **2,527** | **100.0%** | Comprehensive domestic waste stream | Mixed Recyclable / Waste | Municipal Sorting Standard |

---

## 3. Dataset Characteristics & Physical Properties
- **Image Resolution**: Original images are approx. 512 × 384 pixels, captured under controlled indoor lighting with white background poster boards.
- **Color Space**: 3-channel standard RGB.
- **Challenge Factors**:
  - Transparent glass vs. clear plastic bottles (optical similarity).
  - Glossy magazines vs. lightweight cardboard cartons (texture overlap).
  - Crushed metal cans vs. crumpled aluminum foil.
  - Class imbalance: The `trash` class contains only 137 images (5.4%), requiring stratified splitting and balanced metrics.

---

## 4. Partitioning & Leakage Prevention
To guarantee rigorous, reproducible scientific evaluation and prevent data leakage:
- **Stratified Split**: 70% Train, 15% Validation, 15% Held-out Test.
- **Split Breakdown**:
  - **Train**: 1,768 images
  - **Validation**: 379 images
  - **Test**: 380 images
- **Seed**: `42` across all splitting mechanisms.
- The split manifest is saved at `data/splits.csv`.

---

## 5. Automated Data Acquisition & Setup
The repository automates dataset retrieval using `kagglehub`. When running `src/train.py`, the dataset is automatically resolved from cache or downloaded:
```bash
python -c "import kagglehub; kagglehub.dataset_download('asdasdasasdas/garbage-classification')"
```
Alternatively, configure `DATA_DIR` in `.env` to point to any local folder holding the 6 class directories.

---

## 6. Sample Images
A curated subset of representative images (3 per class, 18 total) is included in `data/sample_images/` for instant local testing, offline unit test validation, and UI demos.

---

## 7. Citation
```bibtex
@article{thung2016trashnet,
  title={Classification of Trash for Recyclability Status},
  author={Thung, Gary and Yang, Mindy},
  journal={CS229 Project Report, Stanford University},
  year={2016}
}
```
