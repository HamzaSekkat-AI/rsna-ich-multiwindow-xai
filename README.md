# RSNA ICH Multi-window XAI

Research code for slice-level, multi-label detection of intracranial hemorrhage subtypes from non-contrast head CT using direct DICOM processing, three CT windows, grouped three-fold cross-validation, and complementary explainability analyses.

> **Research-use only.** This software has not been validated for clinical diagnosis, autonomous triage, treatment decisions, or worklist prioritization. Every examination requires complete review by a qualified radiologist.

## What is included

- Direct DICOM-to-Hounsfield-unit conversion
- Brain, subdural, and bone window construction
- Long-to-wide conversion of the binary RSNA labels
- Patient/study-aware grouped three-fold cross-validation
- ImageNet-initialized ResNet34 multi-label classifier
- Weighted binary cross-entropy, AdamW, learning-rate scheduling, and early stopping
- Validation-derived class-specific thresholds
- ROC AUC, PR AUC, precision, sensitivity, specificity, F1, Brier score, and confusion matrices
- Grad-CAM from `layer4[2].conv2`
- Integrated Gradients and occlusion sensitivity
- Input-window ablation
- Deletion/insertion faithfulness analysis
- Fold-specific checkpoints, predictions, metrics, plots, and reports

## Dataset

This repository does **not** redistribute RSNA DICOM images, labels, or patient/study identifiers. Obtain the RSNA 2019 Intracranial Hemorrhage Detection Challenge data independently from its authorized source and comply with its data-use terms.

Expected local layout:

```text
rsna-intracranial-hemorrhage-detection/
├── stage_2_train/
│   ├── ID_....dcm
│   └── ...
└── stage_2_train.csv
```

The label file is expected to contain columns `ID` and `Label`. Labels are binary expert-derived annotations (0/1), not probabilities.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e .
```

Install the PyTorch build matching your CUDA version when GPU acceleration is required.

## Run the complete pipeline

```bash
rsna-ich-xai \
  --dataset_root /path/to/rsna-intracranial-hemorrhage-detection \
  --output_dir outputs_rsna_xai \
  --epochs 30 \
  --num_folds 3 \
  --image_size 256 \
  --batch_size 8 \
  --num_workers 4 \
  --max_xai_samples_per_class 4
```

Equivalent execution without installation:

```bash
PYTHONPATH=src python scripts/run_pipeline.py \
  --dataset_root /path/to/dataset \
  --output_dir outputs_rsna_xai
```

To resume interrupted runs:

```bash
rsna-ich-xai \
  --dataset_root /path/to/dataset \
  --output_dir outputs_rsna_xai \
  --resume_if_available
```

## Outputs

```text
outputs_rsna_xai/
├── config.json
├── tables/
├── metrics/
├── fold_1/
│   ├── checkpoints/
│   ├── metrics/
│   ├── plots/
│   ├── predictions/
│   ├── tables/
│   └── xai/
├── fold_2/
├── fold_3/
└── cv_report_summary.md
```

## Reproducibility notes

- The default random seed is 42.
- Data are grouped by `StudyInstanceUID`, with `PatientID` used as fallback when necessary.
- All available labeled axial slices are included; no lesion-prescreening or slice-selection heuristic is used.
- Resizing is in-plane only, from the native matrix to 256 × 256 using bilinear interpolation. No through-plane resampling is performed.
- The three fold-specific checkpoints are evaluated separately and are not merged or ensembled.
- Thresholds are optimized on the validation partition of each fold and transferred only to the corresponding held-out test partition.
- The best checkpoint is selected using validation macro ROC AUC.

## Important implementation alignment

This release uses **ResNet34** and extracts Grad-CAM from **`layer4[2].conv2`**, matching the revised manuscript. The initially supplied monolithic script contained legacy ResNet50/`conv3` code; those inconsistencies were corrected in this repository release.

## Publication figures

The pipeline exports diagnostic plots at 300 dpi. For final journal production, regenerate or export selected figures at the journal-required resolution (for example, 1200 dpi for line art) without changing the underlying numerical results.

