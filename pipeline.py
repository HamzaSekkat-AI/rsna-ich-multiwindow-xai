#!/usr/bin/env python3
"""
RSNA Intracranial Hemorrhage classification and explainable AI pipeline.

What this script does:
- scans an extracted DICOM training folder and builds a manifest
- converts the RSNA CSV labels from long format to wide multi-label format
- splits data by StudyInstanceUID to reduce leakage
- trains a multi-label ResNet34 on 3 CT windows stacked as channels
- evaluates with ROC AUC, PR AUC, F1, precision, recall, sensitivity,
  specificity, Brier score, confusion matrices, and prediction CSVs
- generates XAI outputs: Grad-CAM, Integrated Gradients, Occlusion maps,
  window ablation importance, and deletion/insertion faithfulness curves
- saves article-friendly figures and a markdown summary report

This is a strong baseline for a slice-level explainable AI paper.
It supports the standard Kaggle folder layout:
    rsna-intracranial-hemorrhage-detection/
        stage_2_train/
        stage_2_test/
        stage_2_train.csv
        stage_2_sample_submission.csv
You can point --dataset_root to that folder and the script will auto-detect the
training DICOM folder and labels CSV.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, StratifiedShuffleSplit

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_STRATIFIED_GROUP_KFOLD = True
except Exception:
    HAS_STRATIFIED_GROUP_KFOLD = False
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from captum.attr import IntegratedGradients, Occlusion
    CAPTUM_AVAILABLE = True
except Exception:
    CAPTUM_AVAILABLE = False


CLASS_NAMES = [
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
    "any",
]

WINDOWS = {
    "brain": (40.0, 80.0),
    "subdural": (80.0, 200.0),
    "bone": (600.0, 2800.0),
}


@dataclass
class Config:
    dataset_root: str = ""
    dicom_dir: str = ""
    labels_csv: str = ""
    output_dir: str = "outputs_rsna_xai"
    manifest_csv: str = ""
    seed: int = 42
    image_size: int = 256
    batch_size: int = 8
    num_workers: int = 4
    epochs: int = 30
    num_folds: int = 3
    resume_if_available: bool = False
    lr: float = 1e-4
    weight_decay: float = 1e-4
    val_size: float = 0.15
    test_size: float = 0.15
    min_lr: float = 1e-6
    early_stopping_patience: int = 6
    amp: bool = True
    pretrained: bool = True
    train_subset: float = 1.0
    max_xai_samples_per_class: int = 8
    deletion_insertion_steps: int = 20
    eval_batch_size: int = 16
    force_manifest_rebuild: bool = False
    train_any_weight: float = 2.0
    gradcam_layer: str = "layer4[2].conv2"
    report_title: str = "RSNA ICH Explainable AI Report"
    save_half_precision_predictions: bool = False
    train_flip_prob: float = 0.5
    positive_threshold_grid: List[float] = field(
        default_factory=lambda: [float(x) for x in np.linspace(0.05, 0.95, 19)]
    )


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="RSNA ICH XAI pipeline")
    parser.add_argument("--dataset_root", type=str, default="",
                        help="Root Kaggle folder containing stage_2_train/ and stage_2_train.csv")
    parser.add_argument("--dicom_dir", type=str, default="",
                        help="Direct path to extracted labeled DICOM training folder. Overrides auto-detection.")
    parser.add_argument("--labels_csv", type=str, default="",
                        help="Path to stage_2_train.csv labels file. Overrides auto-detection.")
    parser.add_argument("--output_dir", type=str, default="outputs_rsna_xai")
    parser.add_argument("--manifest_csv", type=str, default="")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num_folds", type=int, default=3)
    parser.add_argument("--resume_if_available", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_subset", type=float, default=1.0)
    parser.add_argument("--max_xai_samples_per_class", type=int, default=8)
    parser.add_argument("--deletion_insertion_steps", type=int, default=20)
    parser.add_argument("--force_manifest_rebuild", action="store_true")
    args = parser.parse_args()
    cfg = Config(**vars(args))
    return resolve_dataset_paths(cfg)


def resolve_dataset_paths(cfg: Config) -> Config:
    if cfg.dataset_root:
        root = Path(cfg.dataset_root)
        if not root.exists():
            raise FileNotFoundError(f"dataset_root not found: {root}")
        if not cfg.dicom_dir:
            candidate = root / "stage_2_train"
            if candidate.exists():
                cfg.dicom_dir = str(candidate)
        if not cfg.labels_csv:
            for name in ["stage_2_train.csv", "stage_2_train"]:
                candidate = root / name
                if candidate.exists() and candidate.is_file():
                    cfg.labels_csv = str(candidate)
                    break

    if cfg.dicom_dir:
        dicom_path = Path(cfg.dicom_dir)
        if dicom_path.is_dir() and (dicom_path / "stage_2_train").exists() and not list(dicom_path.glob("*.dcm")):
            cfg.dicom_dir = str(dicom_path / "stage_2_train")

    if not cfg.dicom_dir or not Path(cfg.dicom_dir).exists():
        raise FileNotFoundError(
            "Could not find the labeled training DICOM folder. Pass --dataset_root pointing to the Kaggle folder or --dicom_dir directly."
        )
    if not cfg.labels_csv or not Path(cfg.labels_csv).exists():
        raise FileNotFoundError(
            "Could not find the labels CSV. Pass --dataset_root pointing to the Kaggle folder or --labels_csv directly."
        )
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dirs(output_dir: Path) -> Dict[str, Path]:
    dirs = {
        "root": output_dir,
        "checkpoints": output_dir / "checkpoints",
        "metrics": output_dir / "metrics",
        "plots": output_dir / "plots",
        "predictions": output_dir / "predictions",
        "xai": output_dir / "xai",
        "tables": output_dir / "tables",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def save_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dcm_number(value, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (list, tuple)):
        return float(value[0]) if len(value) else float(default)
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        try:
            first = list(value)[0]
            return float(first)
        except Exception:
            return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def build_manifest(cfg: Config, dirs: Dict[str, Path]) -> pd.DataFrame:
    manifest_path = Path(cfg.manifest_csv) if cfg.manifest_csv else dirs["tables"] / "dicom_manifest.csv"
    if manifest_path.exists() and not cfg.force_manifest_rebuild:
        manifest = pd.read_csv(manifest_path)
        return manifest

    dicom_root = Path(cfg.dicom_dir)
    if not dicom_root.exists():
        raise FileNotFoundError(f"DICOM directory not found: {dicom_root}")

    paths = sorted([p for p in dicom_root.rglob("*.dcm") if p.is_file()])
    if not paths:
        raise FileNotFoundError(f"No DICOM files found under: {dicom_root}")

    rows: List[dict] = []
    for path in tqdm(paths, desc="Scanning DICOM headers"):
        try:
            ds = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                force=True,
                specific_tags=[
                    "PatientID",
                    "StudyInstanceUID",
                    "SeriesInstanceUID",
                    "InstanceNumber",
                    "ImagePositionPatient",
                    "SliceLocation",
                    "RescaleSlope",
                    "RescaleIntercept",
                ],
            )
        except Exception:
            continue

        image_id = path.stem
        ipp = getattr(ds, "ImagePositionPatient", None)
        z_val = None
        try:
            if ipp is not None and len(ipp) >= 3:
                z_val = float(ipp[2])
        except Exception:
            z_val = None

        rows.append(
            {
                "image_id": image_id,
                "path": str(path),
                "PatientID": str(getattr(ds, "PatientID", "")),
                "StudyInstanceUID": str(getattr(ds, "StudyInstanceUID", "")),
                "SeriesInstanceUID": str(getattr(ds, "SeriesInstanceUID", "")),
                "InstanceNumber": dcm_number(getattr(ds, "InstanceNumber", None), default=np.nan),
                "SliceLocation": dcm_number(getattr(ds, "SliceLocation", None), default=np.nan),
                "ImagePositionZ": z_val,
                "RescaleSlope": dcm_number(getattr(ds, "RescaleSlope", None), default=1.0),
                "RescaleIntercept": dcm_number(getattr(ds, "RescaleIntercept", None), default=0.0),
            }
        )

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError("Manifest is empty. DICOM scan failed.")

    manifest.to_csv(manifest_path, index=False)
    return manifest


def parse_label_row(value: str) -> Tuple[str, str]:
    base, label = value.rsplit("_", 1)
    return base, label


def read_rsna_labels(labels_csv: str) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv)
    if not {"ID", "Label"}.issubset(set(labels.columns)):
        raise ValueError("Labels CSV must contain columns 'ID' and 'Label'.")

    parsed = labels["ID"].apply(parse_label_row)
    labels["image_id"] = parsed.apply(lambda x: x[0])
    labels["class_name"] = parsed.apply(lambda x: x[1])

    wide = labels.pivot_table(
        index="image_id",
        columns="class_name",
        values="Label",
        aggfunc="first",
    ).reset_index()

    for class_name in CLASS_NAMES:
        if class_name not in wide.columns:
            wide[class_name] = 0.0

    wide = wide[["image_id"] + CLASS_NAMES].copy()
    # RSNA reference labels are binary expert annotations (0/1), not probabilities.
    wide[CLASS_NAMES] = wide[CLASS_NAMES].fillna(0).astype(np.float32)
    return wide


def build_dataset_table(cfg: Config, dirs: Dict[str, Path]) -> pd.DataFrame:
    manifest = build_manifest(cfg, dirs)
    labels = read_rsna_labels(cfg.labels_csv)

    df = manifest.merge(labels, on="image_id", how="inner")
    if df.empty:
        raise RuntimeError(
            "No rows left after merging manifest with labels. Check that your DICOM file names match the label IDs."
        )

    if cfg.train_subset < 1.0:
        df = df.sample(frac=cfg.train_subset, random_state=cfg.seed).reset_index(drop=True)

    split_key = df["StudyInstanceUID"].fillna("")
    split_key = np.where(split_key.astype(str).str.len() > 0, split_key, df["PatientID"].fillna(df["image_id"]))
    df["group_key"] = split_key
    df = df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    df.to_csv(dirs["tables"] / "dataset_table_full.csv", index=False)
    return df


def build_group_targets(full_df: pd.DataFrame) -> pd.DataFrame:
    agg = {c: "max" for c in CLASS_NAMES}
    group_df = full_df.groupby("group_key", as_index=False).agg(agg)
    subtype_cols = [c for c in CLASS_NAMES if c != "any"]
    for c in CLASS_NAMES:
        group_df[c] = group_df[c].fillna(0).astype(float)
    group_df["strat_signature"] = group_df[subtype_cols].round().astype(int).astype(str).agg("".join, axis=1)
    counts = group_df["strat_signature"].value_counts()
    rare = counts[counts < 2].index
    group_df.loc[group_df["strat_signature"].isin(rare), "strat_signature"] = "rare"
    group_df["strat_target"] = np.where(
        group_df["any"].round().astype(int) == 0,
        "neg",
        "pos_" + group_df["strat_signature"].astype(str),
    )
    counts2 = group_df["strat_target"].value_counts()
    rare2 = counts2[counts2 < 2].index
    group_df.loc[group_df["strat_target"].isin(rare2), "strat_target"] = np.where(
        group_df.loc[group_df["strat_target"].isin(rare2), "any"].round().astype(int) == 0,
        "neg",
        "pos_mixed"
    )
    return group_df


def make_outer_group_splits(cfg: Config, full_df: pd.DataFrame):
    group_df = build_group_targets(full_df)
    n_groups = len(group_df)
    if n_groups < cfg.num_folds:
        raise ValueError(f"Not enough unique study groups ({n_groups}) for {cfg.num_folds}-fold CV.")

    strat = group_df["strat_target"].astype(str).values
    min_count = group_df["strat_target"].value_counts().min()
    if min_count >= cfg.num_folds:
        splitter = StratifiedKFold(n_splits=cfg.num_folds, shuffle=True, random_state=cfg.seed)
        splits = list(splitter.split(group_df["group_key"].values, strat))
    else:
        gkf = GroupKFold(n_splits=cfg.num_folds)
        splits = list(gkf.split(full_df, groups=full_df["group_key"].astype(str).values))
        return splits, None

    full_groups = full_df["group_key"].astype(str).values
    all_splits = []
    for train_group_idx, test_group_idx in splits:
        train_groups = set(group_df.iloc[train_group_idx]["group_key"].astype(str).tolist())
        test_groups = set(group_df.iloc[test_group_idx]["group_key"].astype(str).tolist())
        train_idx = np.where(np.isin(full_groups, list(train_groups)))[0]
        test_idx = np.where(np.isin(full_groups, list(test_groups)))[0]
        all_splits.append((train_idx, test_idx))
    return all_splits, group_df


def make_inner_group_split(cfg: Config, train_val_df: pd.DataFrame, fold_index: int):
    group_df = build_group_targets(train_val_df)
    test_size = min(max(cfg.val_size, 0.05), 0.4)
    strat = group_df["strat_target"].astype(str).values
    min_count = group_df["strat_target"].value_counts().min()
    if len(group_df) >= 2 and min_count >= 2:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=cfg.seed + fold_index)
        train_group_idx, val_group_idx = next(sss.split(group_df["group_key"].values, strat))
        train_groups = set(group_df.iloc[train_group_idx]["group_key"].astype(str).tolist())
        val_groups = set(group_df.iloc[val_group_idx]["group_key"].astype(str).tolist())
        group_values = train_val_df["group_key"].astype(str).values
        train_idx = np.where(np.isin(group_values, list(train_groups)))[0]
        val_idx = np.where(np.isin(group_values, list(val_groups)))[0]
        return train_idx, val_idx

    inner_groups = train_val_df["group_key"].astype(str).values
    gss_val = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=cfg.seed + fold_index)
    return next(gss_val.split(train_val_df, groups=inner_groups))


def create_cv_split_table(cfg: Config, full_df: pd.DataFrame, fold_index: int, dirs: Dict[str, Path]) -> pd.DataFrame:
    outer_splits, group_summary = make_outer_group_splits(cfg, full_df)
    train_val_idx, test_idx = outer_splits[fold_index]
    train_val_df = full_df.iloc[train_val_idx].reset_index(drop=True)
    test_df = full_df.iloc[test_idx].reset_index(drop=True)

    unique_inner_groups = train_val_df["group_key"].nunique()
    if unique_inner_groups < 2:
        raise ValueError("Need at least two unique study groups in the training portion to create an internal validation split.")

    train_idx, val_idx = make_inner_group_split(cfg, train_val_df, fold_index)
    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    split_df.to_csv(dirs["tables"] / f"dataset_table_fold_{fold_index + 1}.csv", index=False)

    # save study-level stratification summary for transparency in the paper
    if group_summary is not None:
        group_summary.to_csv(dirs["tables"] / "group_stratification_summary.csv", index=False)
    split_df.groupby("split")[CLASS_NAMES].mean().to_csv(dirs["tables"] / f"fold_{fold_index + 1}_split_label_prevalence.csv")
    return split_df


def dicom_to_hu(path: str) -> np.ndarray:
    ds = pydicom.dcmread(path, force=True)
    image = ds.pixel_array.astype(np.float32)
    slope = dcm_number(getattr(ds, "RescaleSlope", None), default=1.0)
    intercept = dcm_number(getattr(ds, "RescaleIntercept", None), default=0.0)
    image = image * slope + intercept
    return image


def apply_window(image_hu: np.ndarray, center: float, width: float) -> np.ndarray:
    low = center - width / 2.0
    high = center + width / 2.0
    image = np.clip(image_hu, low, high)
    image = (image - low) / max(1e-6, (high - low))
    return image.astype(np.float32)


class RSNAHemorrhageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int, augment: bool = False):
        self.df = df.reset_index(drop=True).copy()
        self.image_size = int(image_size)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def _augment_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if self.augment and random.random() < 0.5:
            x = torch.flip(x, dims=[2])
        return x

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        hu = dicom_to_hu(row["path"])
        channels = [apply_window(hu, *WINDOWS[name]) for name in ["brain", "subdural", "bone"]]
        x = np.stack(channels, axis=0)
        x = torch.from_numpy(x).float()
        x = F.interpolate(
            x.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        x = self._augment_tensor(x)
        y = torch.tensor(row[CLASS_NAMES].values.astype(np.float32), dtype=torch.float32)
        meta = {
            "image_id": row["image_id"],
            "path": row["path"],
            "StudyInstanceUID": row.get("StudyInstanceUID", ""),
        }
        return x, y, meta


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    try:
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
    except AttributeError:
        weights = None
    model = models.resnet34(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def compute_pos_weight(train_df: pd.DataFrame) -> torch.Tensor:
    eps = 1e-6
    positives = train_df[CLASS_NAMES].sum(axis=0).values.astype(np.float32)
    negatives = len(train_df) - positives
    pos_weight = (negatives + eps) / (positives + eps)
    pos_weight[-1] *= 2.0
    return torch.tensor(pos_weight, dtype=torch.float32)


def build_loaders(cfg: Config, split_df: pd.DataFrame) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, RSNAHemorrhageDataset]]:
    train_df = split_df[split_df["split"] == "train"].reset_index(drop=True)
    val_df = split_df[split_df["split"] == "val"].reset_index(drop=True)
    test_df = split_df[split_df["split"] == "test"].reset_index(drop=True)

    datasets = {
        "train": RSNAHemorrhageDataset(train_df, image_size=cfg.image_size, augment=True),
        "val": RSNAHemorrhageDataset(val_df, image_size=cfg.image_size, augment=False),
        "test": RSNAHemorrhageDataset(test_df, image_size=cfg.image_size, augment=False),
    }

    train_loader = DataLoader(
        datasets["train"],
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader, datasets


def batch_to_device(batch, device: torch.device):
    x, y, meta = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    return x, y, meta


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
) -> float:
    model.train()
    running_loss = 0.0
    total = 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        x, y, _ = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = x.size(0)
        running_loss += float(loss.item()) * batch_size
        total += batch_size
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / max(1, total)


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    criterion: Optional[nn.Module],
    device: torch.device,
    use_amp: bool,
) -> Dict[str, np.ndarray]:
    model.eval()
    all_logits: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_ids: List[str] = []
    all_paths: List[str] = []
    running_loss = 0.0
    total = 0

    for batch in tqdm(loader, desc="Eval", leave=False):
        x, y, meta = batch_to_device(batch, device)
        with autocast(enabled=use_amp):
            logits = model(x)
            probs = torch.sigmoid(logits)
            if criterion is not None:
                loss = criterion(logits, y)
                running_loss += float(loss.item()) * x.size(0)
        total += x.size(0)

        all_logits.append(logits.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())
        all_targets.append(y.detach().cpu().numpy())
        all_ids.extend(meta["image_id"])
        all_paths.extend(meta["path"])

    return {
        "loss": running_loss / max(1, total) if criterion is not None else np.nan,
        "logits": np.concatenate(all_logits, axis=0),
        "probs": np.concatenate(all_probs, axis=0),
        "targets": np.concatenate(all_targets, axis=0),
        "image_id": np.array(all_ids),
        "path": np.array(all_paths),
    }


def threshold_search(y_true: np.ndarray, y_prob: np.ndarray, grid: Sequence[float]) -> Dict[str, float]:
    best = {}
    for i, class_name in enumerate(CLASS_NAMES):
        scores = []
        for thr in grid:
            pred = (y_prob[:, i] >= thr).astype(int)
            score = f1_score(y_true[:, i], pred, zero_division=0)
            scores.append((score, thr))
        scores.sort(key=lambda x: (x[0], -abs(x[1] - 0.5)), reverse=True)
        best[class_name] = float(scores[0][1])
    return best


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def safe_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / max(1, (tp + fn))
    specificity = tn / max(1, (tn + fp))
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    brier = float(brier_score_loss(y_true, y_prob))
    return {
        "threshold": float(threshold),
        "roc_auc": safe_auc(y_true, y_prob),
        "pr_auc": safe_ap(y_true, y_prob),
        "precision": float(precision),
        "f1": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "brier": float(brier),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_multilabel(y_true: np.ndarray, y_prob: np.ndarray, thresholds: Dict[str, float]) -> Dict[str, dict]:
    metrics: Dict[str, dict] = {}
    per_class = []
    for i, class_name in enumerate(CLASS_NAMES):
        cls_metrics = binary_metrics(y_true[:, i].astype(int), y_prob[:, i], thresholds[class_name])
        metrics[class_name] = cls_metrics
        per_class.append(cls_metrics)

    metrics["macro"] = {
        key: float(np.nanmean([m[key] for m in per_class]))
        for key in ["roc_auc", "pr_auc", "precision", "f1", "sensitivity", "specificity", "brier"]
    }
    metrics["micro"] = {
        "roc_auc": safe_auc(y_true.reshape(-1), y_prob.reshape(-1)),
        "pr_auc": safe_ap(y_true.reshape(-1), y_prob.reshape(-1)),
        "f1": float(f1_score(y_true.reshape(-1), (y_prob.reshape(-1) >= 0.5).astype(int), zero_division=0)),
    }
    return metrics


def metrics_to_dataframe(metrics: Dict[str, dict]) -> pd.DataFrame:
    rows = []
    for k, v in metrics.items():
        row = {"class": k}
        row.update(v)
        rows.append(row)
    return pd.DataFrame(rows)


def save_predictions(name: str, pred_dict: Dict[str, np.ndarray], dirs: Dict[str, Path]) -> None:
    df = pd.DataFrame({
        "image_id": pred_dict["image_id"],
        "path": pred_dict["path"],
    })
    for i, class_name in enumerate(CLASS_NAMES):
        df[f"target_{class_name}"] = pred_dict["targets"][:, i]
        df[f"prob_{class_name}"] = pred_dict["probs"][:, i]
        df[f"logit_{class_name}"] = pred_dict["logits"][:, i]
    df.to_csv(dirs["predictions"] / f"{name}_predictions.csv", index=False)


def plot_training_history(history: List[dict], save_path: Path) -> None:
    hist = pd.DataFrame(history)
    if hist.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(hist["epoch"], hist["train_loss"], label="train_loss")
    axes[0].plot(hist["epoch"], hist["val_loss"], label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and validation loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if "val_macro_auc" in hist.columns:
        axes[1].plot(hist["epoch"], hist["val_macro_auc"], label="val_macro_auc")
    if "val_macro_pr_auc" in hist.columns:
        axes[1].plot(hist["epoch"], hist["val_macro_pr_auc"], label="val_macro_pr_auc")
    if "val_macro_f1" in hist.columns:
        axes[1].plot(hist["epoch"], hist["val_macro_f1"], label="val_macro_f1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation metrics")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, class_name in enumerate(CLASS_NAMES):
        if len(np.unique(y_true[:, i])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        auc = roc_auc_score(y_true[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, label=f"{class_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_pr_curves(y_true: np.ndarray, y_prob: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, class_name in enumerate(CLASS_NAMES):
        if len(np.unique(y_true[:, i])) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
        ap = average_precision_score(y_true[:, i], y_prob[:, i])
        ax.plot(recall, precision, label=f"{class_name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_confusion_matrices(y_true: np.ndarray, y_prob: np.ndarray, thresholds: Dict[str, float], plots_dir: Path) -> None:
    for i, class_name in enumerate(CLASS_NAMES):
        y_pred = (y_prob[:, i] >= thresholds[class_name]).astype(int)
        cm = confusion_matrix(y_true[:, i], y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["neg", "pos"])
        ax.set_yticklabels(["neg", "pos"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion matrix - {class_name}")
        for (r, c), value in np.ndenumerate(cm):
            ax.text(c, r, int(value), ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(plots_dir / f"confusion_{class_name}.png", dpi=300)
        plt.close(fig)


def build_fold_report(
    cfg: Config,
    history: List[dict],
    val_metrics: Dict[str, dict],
    test_metrics: Dict[str, dict],
    thresholds: Dict[str, float],
    dirs: Dict[str, Path],
    fold_name: str,
) -> None:
    best_epoch = max(history, key=lambda x: x.get("val_macro_auc", float("-inf"))) if history else {"epoch": None}
    macro = test_metrics.get("macro", {})
    lines = [
        f"# {cfg.report_title} - {fold_name}",
        "",
        "## Run summary",
        f"- Best epoch: {best_epoch.get('epoch')}",
        f"- Best validation ROC AUC: {best_epoch.get('val_macro_auc', float('nan')):.6f}" if best_epoch.get("val_macro_auc") is not None else "- Best validation ROC AUC: n/a",
        f"- Best validation loss: {best_epoch.get('val_loss', float('nan')):.6f}" if best_epoch.get("val_loss") is not None else "- Best validation loss: n/a",
        f"- Validation macro ROC AUC: {val_metrics.get('macro', {}).get('roc_auc', float('nan')):.4f}",
        f"- Test macro ROC AUC: {macro.get('roc_auc', float('nan')):.4f}",
        f"- Test macro PR AUC: {macro.get('pr_auc', float('nan')):.4f}",
        f"- Test macro F1: {macro.get('f1', float('nan')):.4f}",
        "",
        "## Decision thresholds",
    ]
    for class_name in CLASS_NAMES:
        lines.append(f"- {class_name}: {thresholds[class_name]:.2f}")
    lines.append("")
    lines.append("## Files")
    lines.append("- metrics/: JSON and CSV metrics")
    lines.append("- plots/: training curves, ROC, PR, confusion matrices")
    lines.append("- xai/: Grad-CAM, Integrated Gradients, Occlusion, faithfulness plots")
    lines.append("- predictions/: per-image probabilities and logits")
    (dirs["root"] / "report_summary.md").write_text("\n".join(lines), encoding="utf-8")


def flatten_metric_dict(metrics: Dict[str, dict], prefix: str) -> dict:
    row = {}
    for class_name, values in metrics.items():
        for key, value in values.items():
            row[f"{prefix}_{class_name}_{key}"] = value
    return row


def aggregate_cv_metrics(fold_metrics: List[dict], save_path: Path) -> pd.DataFrame:
    rows = []
    for fold_idx, metrics in enumerate(fold_metrics, start=1):
        row = {"fold": fold_idx}
        row.update(flatten_metric_dict(metrics, prefix="test"))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(save_path.with_name("cv_fold_metrics_flat.csv"), index=False)

    summary_rows = []
    numeric_cols = [c for c in df.columns if c != "fold"]
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        summary_rows.append({
            "metric": col,
            "mean": float(series.mean()),
            "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(save_path, index=False)
    return summary_df


def build_cv_report(cfg: Config, cv_summary: pd.DataFrame, fold_summaries: List[dict], root_dir: Path) -> None:
    wanted = [
        "test_macro_roc_auc",
        "test_macro_pr_auc",
        "test_macro_f1",
        "test_macro_precision",
        "test_macro_sensitivity",
        "test_macro_specificity",
        "test_macro_brier",
    ]
    lines = [
        f"# {cfg.report_title} - {cfg.num_folds}-fold cross-validation",
        "",
        f"- Epochs: {cfg.epochs}",
        f"- Folds: {cfg.num_folds}",
        f"- Image size: {cfg.image_size}",
        f"- Batch size: {cfg.batch_size}",
        "",
        "## Cross-validation summary (test folds)",
    ]
    cv_summary = cv_summary.set_index("metric")
    for metric in wanted:
        if metric in cv_summary.index:
            row = cv_summary.loc[metric]
            lines.append(f"- {metric}: {row['mean']:.4f} ± {row['std']:.4f}")
    lines.append("")
    lines.append("## Fold directories")
    for fold in fold_summaries:
        lines.append(f"- fold_{fold['fold']}: best_epoch={fold['best_epoch']}, best_val_macro_auc={fold['best_val_macro_auc']:.4f}")
    (root_dir / "cv_report_summary.md").write_text("\n".join(lines), encoding="utf-8")


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handle_fwd = target_layer.register_forward_hook(self._forward_hook)
        self.handle_bwd = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self) -> None:
        self.handle_fwd.remove()
        self.handle_bwd.remove()

    def __call__(self, x: torch.Tensor, class_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach(), logits.detach()


def overlay_heatmap(image_2d: np.ndarray, heatmap_2d: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    image_rgb = np.stack([image_2d, image_2d, image_2d], axis=-1)
    cmap = plt.get_cmap("jet")
    heatmap_rgb = cmap(np.clip(heatmap_2d, 0, 1))[..., :3]
    overlay = (1 - alpha) * image_rgb + alpha * heatmap_rgb
    overlay = np.clip(overlay, 0, 1)
    return overlay


@torch.no_grad()
def channel_ablation_importance(model: nn.Module, x: torch.Tensor, class_idx: int) -> Dict[str, float]:
    base = torch.sigmoid(model(x))[0, class_idx].item()
    names = ["brain", "subdural", "bone"]
    importances = {}
    for channel_idx, channel_name in enumerate(names):
        x_masked = x.clone()
        x_masked[:, channel_idx, :, :] = 0.0
        prob = torch.sigmoid(model(x_masked))[0, class_idx].item()
        importances[channel_name] = float(base - prob)
    return importances


@torch.no_grad()
def deletion_insertion_auc(
    model: nn.Module,
    x: torch.Tensor,
    saliency_2d: np.ndarray,
    class_idx: int,
    steps: int = 20,
) -> Dict[str, object]:
    sal = saliency_2d.copy().astype(np.float32)
    sal = (sal - sal.min()) / max(1e-8, sal.max() - sal.min())
    h, w = sal.shape
    flat_order = np.argsort(-sal.reshape(-1))
    total_pixels = h * w
    step_pixels = max(1, total_pixels // steps)
    fractions = []
    del_probs = []
    ins_probs = []

    baseline = torch.zeros_like(x)
    insertion = baseline.clone()
    deletion = x.clone()

    for k in range(0, total_pixels + 1, step_pixels):
        frac = min(1.0, k / total_pixels)
        fractions.append(frac)

        del_prob = torch.sigmoid(model(deletion))[0, class_idx].item()
        ins_prob = torch.sigmoid(model(insertion))[0, class_idx].item()
        del_probs.append(del_prob)
        ins_probs.append(ins_prob)

        if k == total_pixels:
            break
        idx = flat_order[k : min(total_pixels, k + step_pixels)]
        rr, cc = np.unravel_index(idx, (h, w))
        deletion[:, :, rr, cc] = 0.0
        insertion[:, :, rr, cc] = x[:, :, rr, cc]

    deletion_auc = float(np.trapz(del_probs, fractions))
    insertion_auc = float(np.trapz(ins_probs, fractions))
    return {
        "fractions": fractions,
        "deletion_curve": del_probs,
        "insertion_curve": ins_probs,
        "deletion_auc": deletion_auc,
        "insertion_auc": insertion_auc,
    }


def save_faithfulness_plot(curves: Dict[str, object], save_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curves["fractions"], curves["deletion_curve"], label=f"Deletion AUC={curves['deletion_auc']:.3f}")
    ax.plot(curves["fractions"], curves["insertion_curve"], label=f"Insertion AUC={curves['insertion_auc']:.3f}")
    ax.set_xlabel("Fraction of salient pixels changed")
    ax.set_ylabel("Predicted probability")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


class SingleItemDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int):
        self.base = RSNAHemorrhageDataset(df, image_size=image_size, augment=False)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        return self.base[idx]


def generate_xai_outputs(
    model: nn.Module,
    split_df: pd.DataFrame,
    cfg: Config,
    device: torch.device,
    dirs: Dict[str, Path],
    thresholds: Dict[str, float],
) -> None:
    model.eval()
    xai_root = dirs["xai"]

    test_df = split_df[split_df["split"] == "test"].reset_index(drop=True)
    dataset = SingleItemDataset(test_df, image_size=cfg.image_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # ResNet34: second convolution of the final BasicBlock (layer4[2].conv2).
    target_layer = model.layer4[2].conv2
    gradcam = GradCAM(model, target_layer)

    ig = IntegratedGradients(model) if CAPTUM_AVAILABLE else None
    occ = Occlusion(model) if CAPTUM_AVAILABLE else None

    saved_rows = []

    per_class_counter = {name: 0 for name in CLASS_NAMES}
    max_per_class = cfg.max_xai_samples_per_class

    for batch in tqdm(loader, desc="XAI", leave=False):
        x, y, meta = batch_to_device(batch, device)
        image_id = meta["image_id"][0]
        path = meta["path"][0]

        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
        true_labels = y[0].detach().cpu().numpy().astype(int)

        positive_classes = [i for i, cls in enumerate(CLASS_NAMES) if true_labels[i] == 1]
        if not positive_classes:
            continue

        for class_idx in positive_classes:
            class_name = CLASS_NAMES[class_idx]
            if per_class_counter[class_name] >= max_per_class:
                continue
            if probs[class_idx] < thresholds[class_name]:
                continue

            class_dir = xai_root / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            cam, _ = gradcam(x, class_idx)
            cam_np = cam[0, 0].detach().cpu().numpy()
            base_np = x[0, 0].detach().cpu().numpy()
            overlay_cam = overlay_heatmap(base_np, cam_np)

            ig_np = None
            occ_np = None
            if ig is not None:
                attr_ig = ig.attribute(x, baselines=torch.zeros_like(x), target=class_idx, n_steps=32)
                attr_ig = attr_ig.abs().sum(dim=1, keepdim=True)
                attr_ig = attr_ig - attr_ig.min()
                attr_ig = attr_ig / (attr_ig.max() + 1e-8)
                ig_np = attr_ig[0, 0].detach().cpu().numpy()
            if occ is not None:
                attr_occ = occ.attribute(
                    x,
                    strides=(3, 16, 16),
                    sliding_window_shapes=(3, 32, 32),
                    baselines=0.0,
                    target=class_idx,
                )
                attr_occ = attr_occ.abs().sum(dim=1, keepdim=True)
                attr_occ = attr_occ - attr_occ.min()
                attr_occ = attr_occ / (attr_occ.max() + 1e-8)
                occ_np = attr_occ[0, 0].detach().cpu().numpy()

            window_imp = channel_ablation_importance(model, x, class_idx)
            faith = deletion_insertion_auc(
                model=model,
                x=x,
                saliency_2d=cam_np,
                class_idx=class_idx,
                steps=cfg.deletion_insertion_steps,
            )

            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(2, 4)

            ax0 = fig.add_subplot(gs[0, 0])
            ax0.imshow(x[0, 0].detach().cpu().numpy(), cmap="gray")
            ax0.set_title("Brain window")
            ax0.axis("off")

            ax1 = fig.add_subplot(gs[0, 1])
            ax1.imshow(x[0, 1].detach().cpu().numpy(), cmap="gray")
            ax1.set_title("Subdural window")
            ax1.axis("off")

            ax2 = fig.add_subplot(gs[0, 2])
            ax2.imshow(x[0, 2].detach().cpu().numpy(), cmap="gray")
            ax2.set_title("Bone window")
            ax2.axis("off")

            ax3 = fig.add_subplot(gs[0, 3])
            ax3.imshow(overlay_cam)
            ax3.set_title(f"Grad-CAM\nprob={probs[class_idx]:.3f}")
            ax3.axis("off")

            ax4 = fig.add_subplot(gs[1, 0])
            if ig_np is not None:
                ax4.imshow(overlay_heatmap(base_np, ig_np))
                ax4.set_title("Integrated Gradients")
            else:
                ax4.text(0.5, 0.5, "Captum not installed", ha="center", va="center")
            ax4.axis("off")

            ax5 = fig.add_subplot(gs[1, 1])
            if occ_np is not None:
                ax5.imshow(overlay_heatmap(base_np, occ_np))
                ax5.set_title("Occlusion")
            else:
                ax5.text(0.5, 0.5, "Captum not installed", ha="center", va="center")
            ax5.axis("off")

            ax6 = fig.add_subplot(gs[1, 2])
            imp_names = list(window_imp.keys())
            imp_vals = [window_imp[k] for k in imp_names]
            ax6.bar(imp_names, imp_vals)
            ax6.set_title("Window ablation importance")
            ax6.tick_params(axis="x", rotation=25)
            ax6.grid(True, alpha=0.3)

            ax7 = fig.add_subplot(gs[1, 3])
            ax7.plot(faith["fractions"], faith["deletion_curve"], label="Deletion")
            ax7.plot(faith["fractions"], faith["insertion_curve"], label="Insertion")
            ax7.set_title("Faithfulness")
            ax7.set_xlabel("Fraction changed")
            ax7.set_ylabel("Probability")
            ax7.legend(fontsize=8)
            ax7.grid(True, alpha=0.3)

            fig.suptitle(f"{class_name} | {image_id}")
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            panel_path = class_dir / f"{image_id}_{class_name}_xai_panel.png"
            fig.savefig(panel_path, dpi=300)
            plt.close(fig)

            faith_plot_path = class_dir / f"{image_id}_{class_name}_faithfulness.png"
            save_faithfulness_plot(faith, faith_plot_path, title=f"{class_name} - {image_id}")

            saved_rows.append(
                {
                    "image_id": image_id,
                    "path": path,
                    "class_name": class_name,
                    "probability": float(probs[class_idx]),
                    "threshold": float(thresholds[class_name]),
                    "window_importance_brain": window_imp["brain"],
                    "window_importance_subdural": window_imp["subdural"],
                    "window_importance_bone": window_imp["bone"],
                    "deletion_auc": faith["deletion_auc"],
                    "insertion_auc": faith["insertion_auc"],
                    "panel_path": str(panel_path),
                    "faithfulness_path": str(faith_plot_path),
                }
            )

            per_class_counter[class_name] += 1

        if all(v >= max_per_class for v in per_class_counter.values()):
            break

    gradcam.remove()
    if saved_rows:
        pd.DataFrame(saved_rows).to_csv(xai_root / "xai_summary.csv", index=False)


def save_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[GradScaler],
    epoch: int,
    best_val_auc: float,
    patience_counter: int,
    history: List[dict],
    cfg: Config,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "best_val_auc": best_val_auc,
            "patience_counter": patience_counter,
            "history": history,
            "config": asdict(cfg),
        },
        path,
    )


def load_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
    device: Optional[torch.device] = None,
) -> dict:
    checkpoint = torch.load(path, map_location=device if device is not None else "cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return {
        "start_epoch": int(checkpoint.get("epoch", 0)) + 1,
        "best_val_auc": float(checkpoint.get("best_val_auc", float("-inf"))),
        "patience_counter": int(checkpoint.get("patience_counter", 0)),
        "history": checkpoint.get("history", []),
    }


def maybe_load_completed_fold(dirs: Dict[str, Path]) -> Optional[Tuple[dict, dict]]:
    test_metrics_path = dirs["metrics"] / "test_metrics.json"
    history_path = dirs["tables"] / "training_history.csv"
    if not test_metrics_path.exists():
        return None

    with open(test_metrics_path, "r", encoding="utf-8") as f:
        test_metrics = json.load(f)

    best_epoch = None
    best_val_macro_auc = float("nan")
    if history_path.exists():
        hist = pd.read_csv(history_path)
        if not hist.empty and "val_macro_auc" in hist.columns:
            best_row = hist.loc[hist["val_macro_auc"].astype(float).idxmax()]
            best_epoch = int(best_row["epoch"])
            best_val_macro_auc = float(best_row["val_macro_auc"])

    summary = {
        "best_epoch": best_epoch,
        "best_val_macro_auc": best_val_macro_auc,
    }
    return test_metrics, summary


def train_and_validate(
    cfg: Config,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    dirs: Dict[str, Path],
    device: torch.device,
) -> List[dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
        min_lr=cfg.min_lr,
    )
    scaler = GradScaler(enabled=(cfg.amp and device.type == "cuda"))
    history: List[dict] = []
    best_val_auc = float("-inf")
    patience_counter = 0
    last_path = dirs["checkpoints"] / "last_model.pt"
    best_path = dirs["checkpoints"] / "best_model.pt"
    interrupted_path = dirs["checkpoints"] / "interrupted_model.pt"
    start_epoch = 1

    if cfg.resume_if_available and last_path.exists():
        state = load_training_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        start_epoch = state["start_epoch"]
        best_val_auc = state["best_val_auc"]
        patience_counter = state["patience_counter"]
        history = list(state["history"])
        print(f"Resuming training from epoch {start_epoch} using checkpoint: {last_path}")

    if start_epoch > cfg.epochs:
        print(f"Checkpoint already reached epoch {cfg.epochs}. Skipping training loop.")
        pd.DataFrame(history).to_csv(dirs["tables"] / "training_history.csv", index=False)
        return history

    current_epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, cfg.epochs + 1):
            current_epoch = epoch
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                scaler=scaler,
                use_amp=(cfg.amp and device.type == "cuda"),
            )
            val_pred = predict(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                use_amp=(cfg.amp and device.type == "cuda"),
            )
            val_loss = float(val_pred["loss"])
            scheduler.step(val_loss)

            epoch_thresholds = {class_name: 0.5 for class_name in CLASS_NAMES}
            val_metrics = evaluate_multilabel(val_pred["targets"], val_pred["probs"], epoch_thresholds)
            val_macro_auc = float(val_metrics["macro"]["roc_auc"])
            val_macro_pr_auc = float(val_metrics["macro"]["pr_auc"])
            val_macro_f1 = float(val_metrics["macro"]["f1"])

            improved = val_macro_auc > best_val_auc
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_auc": val_macro_auc,
                "val_macro_pr_auc": val_macro_pr_auc,
                "val_macro_f1": val_macro_f1,
                "lr": optimizer.param_groups[0]["lr"],
                "is_best": int(improved),
            }
            history.append(row)
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | val_macro_auc={val_macro_auc:.5f} | val_macro_f1={val_macro_f1:.5f} | lr={optimizer.param_groups[0]['lr']:.2e}"
            )

            if improved:
                best_val_auc = val_macro_auc
                patience_counter = 0
                save_training_checkpoint(
                    path=best_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_val_auc=best_val_auc,
                    patience_counter=patience_counter,
                    history=history,
                    cfg=cfg,
                )
            else:
                patience_counter += 1

            save_training_checkpoint(
                path=last_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_val_auc=best_val_auc,
                patience_counter=patience_counter,
                history=history,
                cfg=cfg,
            )

            if patience_counter >= cfg.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    except KeyboardInterrupt:
        completed_epoch = history[-1]["epoch"] if history else max(0, current_epoch - 1)
        save_training_checkpoint(
            path=interrupted_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=completed_epoch,
            best_val_auc=best_val_auc,
            patience_counter=patience_counter,
            history=history,
            cfg=cfg,
        )
        save_training_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=completed_epoch,
            best_val_auc=best_val_auc,
            patience_counter=patience_counter,
            history=history,
            cfg=cfg,
        )
        pd.DataFrame(history).to_csv(dirs["tables"] / "training_history.csv", index=False)
        print(f"Training interrupted. Saved resume checkpoint to: {last_path}")
        raise

    pd.DataFrame(history).to_csv(dirs["tables"] / "training_history.csv", index=False)
    return history


def load_best_model(model: nn.Module, checkpoint_path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def main(cfg: Config) -> None:
    seed_everything(cfg.seed)
    output_dir = Path(cfg.output_dir)
    root_dirs = ensure_dirs(output_dir)
    save_json(asdict(cfg), root_dirs["root"] / "config.json")
    device = get_device()
    print(f"Using device: {device}")

    full_df = build_dataset_table(cfg, root_dirs)
    fold_test_metrics: List[dict] = []
    fold_summaries: List[dict] = []

    for fold_index in range(cfg.num_folds):
        fold_name = f"fold_{fold_index + 1}"
        fold_root = root_dirs["root"] / fold_name
        dirs = ensure_dirs(fold_root)
        print(f"\n===== Starting {fold_name} / {cfg.num_folds} =====")

        if cfg.resume_if_available:
            completed = maybe_load_completed_fold(dirs)
            if completed is not None:
                test_metrics, summary = completed
                fold_test_metrics.append(test_metrics)
                fold_summaries.append({
                    "fold": fold_index + 1,
                    "best_epoch": summary.get("best_epoch"),
                    "best_val_macro_auc": float(summary.get("best_val_macro_auc", float("nan"))),
                })
                print(f"Skipping {fold_name}; completed results already exist in {dirs['root']}")
                continue

        split_df = create_cv_split_table(cfg, full_df, fold_index, root_dirs)
        split_df.to_csv(dirs["tables"] / "dataset_table_with_splits.csv", index=False)
        train_loader, val_loader, test_loader, datasets = build_loaders(cfg, split_df)

        train_df = split_df[split_df["split"] == "train"].reset_index(drop=True)
        pos_weight = compute_pos_weight(train_df).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        model = build_model(num_classes=len(CLASS_NAMES), pretrained=cfg.pretrained).to(device)

        try:
            history = train_and_validate(
                cfg=cfg,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                dirs=dirs,
                device=device,
            )
        except KeyboardInterrupt:
            print("Stopped safely. Resume later by re-running the same command with --resume_if_available and the same --output_dir.")
            return

        checkpoint_path = dirs["checkpoints"] / "best_model.pt"
        model = load_best_model(model, checkpoint_path, device)

        val_pred = predict(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=(cfg.amp and device.type == "cuda"),
        )
        test_pred = predict(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            use_amp=(cfg.amp and device.type == "cuda"),
        )

        save_predictions("val", val_pred, dirs)
        save_predictions("test", test_pred, dirs)

        thresholds = threshold_search(val_pred["targets"], val_pred["probs"], cfg.positive_threshold_grid)
        save_json(thresholds, dirs["metrics"] / "thresholds.json")

        val_metrics = evaluate_multilabel(val_pred["targets"], val_pred["probs"], thresholds)
        test_metrics = evaluate_multilabel(test_pred["targets"], test_pred["probs"], thresholds)

        save_json(val_metrics, dirs["metrics"] / "val_metrics.json")
        save_json(test_metrics, dirs["metrics"] / "test_metrics.json")

        metrics_to_dataframe(val_metrics).to_csv(dirs["metrics"] / "val_metrics.csv", index=False)
        metrics_to_dataframe(test_metrics).to_csv(dirs["metrics"] / "test_metrics.csv", index=False)

        plot_training_history(history, dirs["plots"] / "training_history.png")
        plot_roc_curves(test_pred["targets"], test_pred["probs"], dirs["plots"] / "roc_curves_test.png")
        plot_pr_curves(test_pred["targets"], test_pred["probs"], dirs["plots"] / "pr_curves_test.png")
        plot_confusion_matrices(test_pred["targets"], test_pred["probs"], thresholds, dirs["plots"])

        generate_xai_outputs(
            model=model,
            split_df=split_df,
            cfg=cfg,
            device=device,
            dirs=dirs,
            thresholds=thresholds,
        )

        build_fold_report(
            cfg=cfg,
            history=history,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            thresholds=thresholds,
            dirs=dirs,
            fold_name=fold_name,
        )

        best_epoch_row = max(history, key=lambda x: x.get("val_macro_auc", float("-inf"))) if history else {"epoch": None, "val_macro_auc": float("nan")}
        fold_summaries.append({
            "fold": fold_index + 1,
            "best_epoch": best_epoch_row.get("epoch"),
            "best_val_macro_auc": float(best_epoch_row.get("val_macro_auc", float("nan"))),
        })
        fold_test_metrics.append(test_metrics)

        print(f"Completed {fold_name}. Results saved to: {dirs['root']}")

    if fold_test_metrics:
        cv_summary = aggregate_cv_metrics(fold_test_metrics, root_dirs["metrics"] / "cv_metrics_summary.csv")
        pd.DataFrame(fold_summaries).to_csv(root_dirs["metrics"] / "fold_training_summary.csv", index=False)
        build_cv_report(cfg, cv_summary, fold_summaries, root_dirs["root"])
        print(f"\nDone. Cross-validation results saved to: {root_dirs['root']}")


if __name__ == "__main__":
    config = parse_args()
    main(config)
