# -*- coding: utf-8 -*-
"""
TransAttUNet.py

Colab 노트북(TransAttUNet.ipynb) 내용을 .py로 정리한 버전.
- TransUNet (R50-ViT-B_16) 학습/평가
- Dynamic sampling(positive_ratio), heavy aug, combo loss, triplet post-process 옵션 포함
- Test set에서 ViT attention 시각화 저장(옵션)
"""

import sys
sys.path.insert(0, "./TransUNet")

import os
import glob
import json
import time
import math
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset

import segmentation_models_pytorch as smp
from sklearn.model_selection import StratifiedKFold
from scipy import ndimage

# Albumentations
from albumentations import (
    HorizontalFlip,
    ShiftScaleRotate,
    Normalize,
    Resize,
    Compose,
    RandomBrightnessContrast,
    RandomGamma,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
)

try:
    from albumentations.pytorch import ToTensorV2
    ToTensorTransform = ToTensorV2
except ImportError:
    try:
        from albumentations.torch import ToTensor
        ToTensorTransform = ToTensor
    except ImportError:
        class ToTensorTransform:
            def __call__(self, **kwargs):
                image = kwargs["image"]
                mask = kwargs["mask"]
                if len(image.shape) == 3:
                    image = image.transpose(2, 0, 1)
                image = torch.from_numpy(image).float()
                mask = torch.from_numpy(mask).unsqueeze(0).float()
                return {"image": image, "mask": mask}


warnings.filterwarnings("ignore")

# =========================
# 0) TransUNet import
# =========================
try:
    # 논문 구현 그대로의 R50 + ViT + skip 구조 사용
    from networks.vit_seg_modeling_attn2 import VisionTransformer as ViT_seg, CONFIGS as CONFIGS_ViT_seg
    print("✓ TransUNet (vit_seg_modeling_attn2) imported successfully")
except Exception as e:
    print(f"❌ Import error: {e}")
    raise


print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)


# =========================
# 1) Config
# =========================
@dataclass
class Config:
    experiment_name: str = "baseline"

    # paths
    data_path: Path = Path("./siim_dataset/train_png")
    test_path: Path = Path("./siim_dataset/test_png")
    rle_path: Path = Path("./siim_dataset/train-rle.csv")
    base_save_dir: Path = Path("./ablation_results")

    # Data Sampling - positive_ratio
    use_dynamic_sampling: bool = False
    initial_positive_ratio: float = 0.8
    final_positive_ratio: float = 0.4

    # Augmentation
    use_heavy_augmentation: bool = False

    # Model
    model_architecture: str = "TransUNet"
    backbone: str = "resnet34"

    # TransUNet params
    vit_name: str = "R50-ViT-B_16"
    vit_patches_size: int = 16
    n_skip: int = 3

    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_metric: str = "dice"
    early_stopping_min_delta: float = 0.0001

    # Loss
    use_combo_loss: bool = False
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    focal_weight: float = 1.0
    focal_alpha: float = 10.0
    focal_gamma: float = 2.0

    # Post-processing
    use_triplet_threshold: bool = False
    triplet_top: float = 0.75
    triplet_bottom: float = 0.3
    triplet_min_area: int = 2000

    # Progressive Training
    use_progressive_training: bool = False
    warmup_epochs: int = 10
    high_sr_epochs: int = 10
    low_sr_epochs: int = 10
    final_epochs: int = 10

    # Progressive Resolution
    use_progressive_resolution: bool = False
    initial_size: int = 512
    final_size: int = 1024
    freeze_encoder_epochs: int = 5

    train_size: int = 512
    original_size: int = 1024

    # Training params
    fold: int = 0
    total_folds: int = 5
    batch_size: int = 4
    accumulation_steps: int = 8
    num_epochs: int = 40
    base_lr: float = 1e-4
    scheduler_type: str = "ReduceLROnPlateau"
    scheduler_patience: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Image params
    image_size: int = 512
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    num_workers: int = 0

    def __post_init__(self):
        if self.use_progressive_training:
            self.num_epochs = self.warmup_epochs + self.high_sr_epochs + self.low_sr_epochs + self.final_epochs

        self._save_dir = Path(self.base_save_dir) / self.experiment_name
        self._save_dir.mkdir(exist_ok=True, parents=True)

    @property
    def save_dir(self) -> Path:
        if not hasattr(self, "_save_dir"):
            self._save_dir = Path(self.base_save_dir) / self.experiment_name
            self._save_dir.mkdir(exist_ok=True, parents=True)
        return self._save_dir

    def get_current_positive_ratio(self, epoch: int) -> float:
        if not self.use_dynamic_sampling:
            return self.initial_positive_ratio

        progress = epoch / max(1, self.num_epochs - 1)
        ratio = self.initial_positive_ratio - ((self.initial_positive_ratio - self.final_positive_ratio) * progress)
        return max(min(ratio, 1.0), 0.0)

    def get_current_image_size(self, epoch: int) -> int:
        return self.train_size

    def should_freeze_encoder(self, epoch: int) -> bool:
        if not self.use_progressive_resolution:
            return False
        transition_epoch = self.num_epochs // 2
        return transition_epoch <= epoch < (transition_epoch + self.freeze_encoder_epochs)

    def get_lr_and_scheduler(self, epoch: int, optimizer) -> Tuple[float, str]:
        if not self.use_progressive_training:
            return self.base_lr, self.scheduler_type

        if epoch < self.warmup_epochs:
            lr = 1e-3
            scheduler_type = "ReduceLROnPlateau"
        elif epoch < self.warmup_epochs + self.high_sr_epochs:
            lr = 1e-5
            scheduler_type = "CosineAnnealingLR"
        elif epoch < self.warmup_epochs + self.high_sr_epochs + self.low_sr_epochs:
            lr = 1e-5
            scheduler_type = "CosineAnnealingLR"
        else:
            lr = 1e-6
            scheduler_type = "constant"

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr, scheduler_type

    def save(self):
        config_dict = {
            "experiment_name": self.experiment_name,
            "data_path": str(self.data_path),
            "test_path": str(self.test_path),
            "rle_path": str(self.rle_path),
            "base_save_dir": str(self.base_save_dir),
            "save_dir": str(self.save_dir),
            "use_dynamic_sampling": self.use_dynamic_sampling,
            "initial_positive_ratio": self.initial_positive_ratio,
            "final_positive_ratio": self.final_positive_ratio,
            "use_heavy_augmentation": self.use_heavy_augmentation,
            "use_combo_loss": self.use_combo_loss,
            "bce_weight": self.bce_weight,
            "dice_weight": self.dice_weight,
            "focal_weight": self.focal_weight,
            "focal_alpha": self.focal_alpha,
            "focal_gamma": self.focal_gamma,
            "use_triplet_threshold": self.use_triplet_threshold,
            "triplet_top": self.triplet_top,
            "triplet_bottom": self.triplet_bottom,
            "triplet_min_area": self.triplet_min_area,
            "use_progressive_training": self.use_progressive_training,
            "warmup_epochs": self.warmup_epochs,
            "high_sr_epochs": self.high_sr_epochs,
            "low_sr_epochs": self.low_sr_epochs,
            "final_epochs": self.final_epochs,
            "use_progressive_resolution": self.use_progressive_resolution,
            "initial_size": self.initial_size,
            "final_size": self.final_size,
            "freeze_encoder_epochs": self.freeze_encoder_epochs,
            "backbone": self.backbone,
            "fold": self.fold,
            "total_folds": self.total_folds,
            "batch_size": self.batch_size,
            "accumulation_steps": self.accumulation_steps,
            "num_epochs": self.num_epochs,
            "base_lr": self.base_lr,
            "scheduler_type": self.scheduler_type,
            "scheduler_patience": self.scheduler_patience,
            "device": self.device,
            "image_size": self.image_size,
            "mean": list(self.mean),
            "std": list(self.std),
            "num_workers": self.num_workers,
        }
        with open(self.save_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=4)


# =========================
# 2) RLE utils
# =========================
def run_length_decode(rle: str, height=1024, width=1024, fill_value=1) -> np.ndarray:
    component = np.zeros((height, width), np.float32).reshape(-1)
    rle = np.array([int(s) for s in rle.strip().split(" ")])
    rle = rle.reshape(-1, 2)
    start = 0
    for index, length in rle:
        start = start + index
        end = start + length
        component[start:end] = fill_value
        start = end
    component = component.reshape(width, height).T
    return component


def run_length_encode(component: np.ndarray) -> str:
    component = component.T.flatten()
    start = np.where(component[1:] > component[:-1])[0] + 1
    end = np.where(component[:-1] > component[1:])[0] + 1
    length = end - start
    rle = []
    for i in range(len(length)):
        if i == 0:
            rle.extend([start[0], length[0]])
        else:
            rle.extend([start[i] - end[i - 1], length[i]])
    return " ".join([str(r) for r in rle])


def build_filename_mapping(data_folder: str, additional_folder: Optional[str] = None) -> Dict[str, str]:
    folders = [data_folder]
    if additional_folder:
        folders.append(additional_folder)

    mapping = {}
    for folder in folders:
        all_files = glob.glob(os.path.join(folder, "*.png"))
        for filepath in all_files:
            filename = os.path.basename(filepath)
            name_without_ext = filename.replace(".png", "")
            name_without_dcm = name_without_ext.replace(".dcm", "")
            mapping[name_without_ext] = filepath
            mapping[name_without_dcm] = filepath
    return mapping


# =========================
# 3) Augmentations
# =========================
def get_transforms(phase: str, size: int, mean, std, use_heavy_aug=False):
    list_transforms = []
    if phase == "train":
        list_transforms.append(HorizontalFlip(p=0.5))

        if use_heavy_aug:
            list_transforms.extend(
                [
                    RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                    RandomGamma(gamma_limit=(80, 120), p=0.3),
                    ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
                    GridDistortion(p=0.3),
                    OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.3),
                    ShiftScaleRotate(
                        shift_limit=0.1,
                        scale_limit=0.1,
                        rotate_limit=15,
                        border_mode=cv2.BORDER_CONSTANT,
                        p=0.5,
                    ),
                ]
            )
        else:
            list_transforms.append(
                ShiftScaleRotate(
                    shift_limit=0,
                    scale_limit=0.1,
                    rotate_limit=10,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                )
            )

    list_transforms.extend(
        [
            Resize(size, size),
            Normalize(mean=mean, std=std, p=1),
            ToTensorTransform(),
        ]
    )
    return Compose(list_transforms)


# =========================
# 4) Dataset + Provider
# =========================
class SIIMDataset(Dataset):
    def __init__(
        self,
        df_all: pd.DataFrame,
        fnames: np.ndarray,
        data_folder: str,
        size: int,
        mean,
        std,
        phase: str,
        test_data_folder: Optional[str] = None,
        use_heavy_aug: bool = False,
    ):
        self.df_all = df_all
        self.root = data_folder
        self.size = size
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms(phase, size, mean, std, use_heavy_aug)

        self.gb = self.df_all.groupby("ImageId")
        self.fnames = fnames

        self.file_mapping = build_filename_mapping(data_folder, test_data_folder)

        # mask area cache (curriculum 등에 사용 가능)
        self.mask_area_cache = []
        for image_id in self.fnames:
            df_g = self.gb.get_group(image_id)
            annotations = df_g[" EncodedPixels"].tolist()
            if annotations[0] == " -1":
                area = 0
            else:
                mask = np.zeros([1024, 1024], dtype=np.float32)
                for rle in annotations:
                    mask += run_length_decode(rle).astype(np.float32)
                area = mask.sum()
            self.mask_area_cache.append(area)

    def __getitem__(self, idx):
        image_id = self.fnames[idx]
        df_g = self.gb.get_group(image_id)
        annotations = df_g[" EncodedPixels"].tolist()

        if image_id not in self.file_mapping:
            raise FileNotFoundError(f"ImageId not in mapping: {image_id}")
        image_path = self.file_mapping[image_id]

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read: {image_path}")

        mask = np.zeros([1024, 1024], dtype=np.float32)
        if annotations[0] != " -1":
            for rle in annotations:
                mask += run_length_decode(rle)
        mask = (mask >= 1).astype("float32")

        augmented = self.transforms(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        return image, mask

    def __len__(self):
        return len(self.fnames)


def provider(
    fold: int,
    total_folds: int,
    data_folder: str,
    df_path: str,
    phase: str,
    size: int,
    mean=None,
    std=None,
    batch_size: int = 8,
    num_workers: int = 0,
    test_data_folder: Optional[str] = None,
    positive_ratio: float = 0.8,
    use_heavy_aug: bool = False,
):
    """
    DataLoader 생성
    positive_ratio: 전체 샘플 중 positive 샘플의 비율
    """
    df_all = pd.read_csv(df_path)
    df = df_all.drop_duplicates("ImageId")

    # Sampling
    df_with_mask = df[df[" EncodedPixels"] != " -1"]
    df_without_mask = df[df[" EncodedPixels"] == " -1"]

    n_positive = len(df_with_mask)
    n_negative = int(n_positive * (1 - positive_ratio) / positive_ratio)

    df_without_mask_sampled = df_without_mask.sample(min(n_negative, len(df_without_mask)), random_state=69)
    df = pd.concat([df_with_mask, df_without_mask_sampled])

    df["has_mask"] = (df[" EncodedPixels"] != " -1").astype(int)

    # K-Fold split
    kfold = StratifiedKFold(total_folds, shuffle=True, random_state=69)
    train_idx, val_idx = list(kfold.split(df["ImageId"], df["has_mask"]))[fold]
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    df_used = train_df if phase == "train" else val_df
    fnames = df_used["ImageId"].values

    dataset = SIIMDataset(
        df_all=df_all,
        fnames=fnames,
        data_folder=data_folder,
        size=size,
        mean=mean,
        std=std,
        phase=phase,
        test_data_folder=test_data_folder,
        use_heavy_aug=use_heavy_aug,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=(phase == "train"),
    )
    return dataloader


# =========================
# 5) Losses
# =========================
def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if target.size() != input.size():
            raise ValueError("Target size must be the same as input size")

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()


class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.alpha * self.focal(input, target) - torch.log(dice_loss(input, target))
        return loss.mean()


class ComboLoss(nn.Module):
    """BCE + Dice + Focal 가중 조합"""
    def __init__(self, bce_weight=1.0, dice_weight=1.0, focal_weight=1.0, focal_alpha=10.0, focal_gamma=2.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss(focal_gamma)
        self.focal_alpha = focal_alpha

    def forward(self, input, target):
        bce = self.bce(input, target)
        dice = 1 - dice_loss(input, target)
        focal = self.focal(input, target)
        return (self.bce_weight * bce + self.dice_weight * dice + self.focal_weight * focal)


# =========================
# 6) Post-processing
# =========================
def triplet_threshold_post_process(prob, top_th=0.75, bottom_th=0.3, min_area=2000):
    if prob.max() < top_th:
        return np.zeros_like(prob)

    seeds = prob > top_th
    expansion_mask = prob > bottom_th

    labeled, num_features = ndimage.label(seeds)
    final_mask = np.zeros_like(prob)

    for i in range(1, num_features + 1):
        component = (labeled == i)
        expanded = ndimage.binary_dilation(component, iterations=20, mask=expansion_mask)
        if expanded.sum() >= min_area:
            final_mask = np.maximum(final_mask, expanded.astype(float))

    return final_mask


def apply_post_processing(probs: torch.Tensor, config: Config):
    if not config.use_triplet_threshold:
        return (probs > 0.5).float()

    batch_size = probs.shape[0]
    processed = torch.zeros_like(probs)
    for i in range(batch_size):
        prob = probs[i, 0].cpu().numpy()
        mask = triplet_threshold_post_process(
            prob,
            top_th=config.triplet_top,
            bottom_th=config.triplet_bottom,
            min_area=config.triplet_min_area,
        )
        processed[i, 0] = torch.from_numpy(mask).float()
    return processed


# =========================
# 7) Metrics
# =========================
def predict(X, threshold):
    preds = (np.copy(X) > threshold).astype("uint8")
    return preds


def metric(probability, truth, threshold=0.5):
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert probability.shape == truth.shape

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)

        neg_index = torch.nonzero(t_sum == 0).squeeze(1)
        pos_index = torch.nonzero(t_sum >= 1).squeeze(1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1) + 1e-8)

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice, dice_neg, dice_pos, num_neg, num_pos


def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    pred[label == ignore_index] = 0
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]


def compute_iou_batch(outputs, labels, classes=None):
    ious = []
    preds = np.copy(outputs)
    labels = np.array(labels)
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    return np.nanmean(ious)


class Meter:
    def __init__(self, phase, epoch):
        self.base_threshold = 0.5
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []

    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        dice, dice_neg, dice_pos, _, _ = metric(probs, targets, self.base_threshold)

        self.base_dice_scores.extend(dice.cpu().numpy() if torch.is_tensor(dice) else [dice])
        self.dice_pos_scores.extend(dice_pos.cpu().numpy() if len(dice_pos) > 0 else [])
        self.dice_neg_scores.extend(dice_neg.cpu().numpy() if len(dice_neg) > 0 else [])

        preds = predict(probs.cpu().numpy(), self.base_threshold)
        iou = compute_iou_batch(preds, targets.cpu().numpy(), classes=[1])
        self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.nanmean(self.base_dice_scores) if len(self.base_dice_scores) else 0.0
        dice_neg = np.nanmean(self.dice_neg_scores) if len(self.dice_neg_scores) else 0.0
        dice_pos = np.nanmean(self.dice_pos_scores) if len(self.dice_pos_scores) else 0.0
        iou = np.nanmean(self.iou_scores) if len(self.iou_scores) else 0.0
        return [dice, dice_neg, dice_pos], iou


def epoch_log(phase, epoch, epoch_loss, meter, start):
    dices, iou = meter.get_metrics()
    dice, dice_neg, dice_pos = dices
    print(
        "Loss: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f | IoU: %0.4f"
        % (epoch_loss, dice, dice_neg, dice_pos, iou)
    )
    return dice, iou


# =========================
# 8) Trainer
# =========================
class Trainer:
    def __init__(self, model, config: Config):
        self.config = config
        self.fold = config.fold
        self.total_folds = config.total_folds
        self.num_workers = config.num_workers

        self.batch_size = {"train": config.batch_size, "val": config.batch_size}
        self.accumulation_steps = config.accumulation_steps
        self.num_epochs = config.num_epochs

        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        self.device = torch.device(config.device)

        self.best_metric = None
        self.early_stop_counter = 0
        self.best_ckpt_metric = None

        self.net = model

        if config.use_combo_loss:
            self.criterion = ComboLoss(
                bce_weight=config.bce_weight,
                dice_weight=config.dice_weight,
                focal_weight=config.focal_weight,
                focal_alpha=config.focal_alpha,
                focal_gamma=config.focal_gamma,
            )
            print(f"✓ Using Combo Loss (BCE:{config.bce_weight}, Dice:{config.dice_weight}, Focal:{config.focal_weight})")
        else:
            self.criterion = MixedLoss(config.focal_alpha, config.focal_gamma)
            print(f"✓ Using Mixed Loss (Focal alpha={config.focal_alpha}, gamma={config.focal_gamma})")

        self.optimizer = optim.Adam(self.net.parameters(), lr=config.base_lr)

        if config.scheduler_type == "ReduceLROnPlateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=config.scheduler_patience,
            )
        elif config.scheduler_type == "CosineAnnealingLR":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.num_epochs)
        else:
            self.scheduler = None

        self.net = self.net.to(self.device)
        if torch.cuda.is_available():
            cudnn.benchmark = True

        self.current_image_size = config.train_size
        self.current_positive_ratio = config.initial_positive_ratio
        self._create_dataloaders()

        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}

    def _create_dataloaders(self):
        self.dataloaders = {
            phase: provider(
                fold=self.fold,
                total_folds=self.total_folds,
                data_folder=str(self.config.data_path),
                df_path=str(self.config.rle_path),
                phase=phase,
                size=self.current_image_size,
                mean=self.config.mean,
                std=self.config.std,
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
                test_data_folder=str(self.config.test_path),
                positive_ratio=self.current_positive_ratio,
                use_heavy_aug=self.config.use_heavy_augmentation,
            )
            for phase in self.phases
        }

    def _update_epoch_settings(self, epoch):
        updated = False

        new_size = self.config.get_current_image_size(epoch)
        if new_size != self.current_image_size:
            print("\n" + "=" * 80)
            print(f"🔍 Resolution Change: {self.current_image_size} → {new_size}")
            print("=" * 80)
            self.current_image_size = new_size
            updated = True

        new_ratio = self.config.get_current_positive_ratio(epoch)
        if abs(new_ratio - self.current_positive_ratio) > 0.01:
            print("\n" + "=" * 80)
            print(f"📊 Positive Sampling Ratio: {self.current_positive_ratio:.2f} → {new_ratio:.2f}")
            print("=" * 80)
            self.current_positive_ratio = new_ratio
            updated = True

        if updated:
            self._create_dataloaders()

        if self.config.should_freeze_encoder(epoch):
            if not hasattr(self, "_encoder_frozen") or not self._encoder_frozen:
                print("\n" + "=" * 80)
                print("🔒 Freezing Encoder")
                print("=" * 80)
                for param in self.net.encoder.parameters():
                    param.requires_grad = False
                self._encoder_frozen = True
        else:
            if hasattr(self, "_encoder_frozen") and self._encoder_frozen:
                print("\n" + "=" * 80)
                print("🔓 Unfreezing Encoder")
                print("=" * 80)
                for param in self.net.encoder.parameters():
                    param.requires_grad = True
                self._encoder_frozen = False

        if self.config.use_progressive_training:
            lr, scheduler_type = self.config.get_lr_and_scheduler(epoch, self.optimizer)
            if scheduler_type != self.config.scheduler_type:
                if scheduler_type == "CosineAnnealingLR":
                    remaining_epochs = self.num_epochs - epoch
                    self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=remaining_epochs)
                elif scheduler_type == "ReduceLROnPlateau":
                    self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=3)
                else:
                    self.scheduler = None
                self.config.scheduler_type = scheduler_type
            print(f"📈 LR: {lr:.2e}, Scheduler: {scheduler_type}")

    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)
        return loss, outputs

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")

        is_train = (phase == "train")
        self.net.train(is_train)
        dataloader = self.dataloaders[phase]

        running_loss = 0.0
        total_batches = len(dataloader)

        if is_train:
            self.optimizer.zero_grad()

        tk0 = tqdm(dataloader, total=total_batches, desc=f"Epoch {epoch} [{phase}]")
        for itr, batch in enumerate(tk0):
            images, targets = batch

            if is_train:
                loss, outputs = self.forward(images, targets)
                loss = loss / self.accumulation_steps
                loss.backward()

                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            else:
                with torch.no_grad():
                    loss, outputs = self.forward(images, targets)

            running_loss += loss.item()

            with torch.no_grad():
                outputs_resized = F.interpolate(
                    outputs.detach(),
                    size=(self.config.original_size, self.config.original_size),
                    mode="bilinear",
                    align_corners=False,
                )
                targets_resized = F.interpolate(
                    targets.detach(),
                    size=(self.config.original_size, self.config.original_size),
                    mode="nearest",
                )
                meter.update(targets_resized.cpu(), outputs_resized.cpu())

            tk0.set_postfix(loss=(running_loss / (itr + 1)))

        if is_train:
            epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        else:
            epoch_loss = running_loss / total_batches

        dice, iou = epoch_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)

        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        for epoch in range(self.num_epochs):
            self._update_epoch_settings(epoch)

            train_loss = self.iterate(epoch, "train")
            val_loss = self.iterate(epoch, "val")

            if val_loss < self.best_loss:
                self.best_loss = val_loss

            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }

            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # checkpoint 기준
            metric_name = self.config.early_stopping_metric
            if metric_name == "dice":
                current_ckpt_metric = self.dice_scores["val"][-1]
            elif metric_name == "iou":
                current_ckpt_metric = self.iou_scores["val"][-1]
            elif metric_name == "loss":
                current_ckpt_metric = -val_loss
            else:
                current_ckpt_metric = self.dice_scores["val"][-1]

            if (self.best_ckpt_metric is None) or (current_ckpt_metric > self.best_ckpt_metric):
                self.best_ckpt_metric = current_ckpt_metric
                print("******** New optimal found, saving state ********")
                state["best_metric"] = self.best_ckpt_metric
                torch.save(state, str(self.config.save_dir / "model.pth"))

            # early stopping
            if self.config.use_early_stopping:
                if metric_name == "dice":
                    current_metric = self.dice_scores["val"][-1]
                    improved = (self.best_metric is None) or (current_metric - self.best_metric > self.config.early_stopping_min_delta)
                elif metric_name == "iou":
                    current_metric = self.iou_scores["val"][-1]
                    improved = (self.best_metric is None) or (current_metric - self.best_metric > self.config.early_stopping_min_delta)
                elif metric_name == "loss":
                    current_metric = val_loss
                    improved = (self.best_metric is None) or (self.best_metric - current_metric > self.config.early_stopping_min_delta)
                else:
                    current_metric = self.dice_scores["val"][-1]
                    improved = (self.best_metric is None) or (current_metric - self.best_metric > self.config.early_stopping_min_delta)

                if improved:
                    self.best_metric = current_metric
                    self.early_stop_counter = 0
                    print(f"[EarlyStopping] {metric_name} improved to {current_metric:.6f}")
                else:
                    self.early_stop_counter += 1
                    print(f"[EarlyStopping] No improvement in {metric_name} for {self.early_stop_counter}/{self.config.early_stopping_patience} epochs")

                if self.early_stop_counter >= self.config.early_stopping_patience:
                    print(f"\n[EarlyStopping] Patience reached. Stopping at epoch {epoch}.")
                    break

            print()


# =========================
# 9) Model create
# =========================
def create_model(config: Config):
    print(f"Creating model with backbone: {config.backbone}")
    architecture = getattr(config, "model_architecture", "Unet")

    try:
        if architecture == "Unet":
            model = smp.Unet(encoder_name=config.backbone, encoder_weights="imagenet", activation=None)
            print(f"✓ Created {architecture} with {config.backbone}")
            return model

        if architecture == "TransUNet":
            vit_name = config.vit_name
            vit_config = CONFIGS_ViT_seg[vit_name]
            vit_config.n_classes = 1
            vit_config.n_skip = config.n_skip

            if vit_name.find("R50") != -1:
                vit_config.patches.grid = (
                    int(config.image_size / config.vit_patches_size),
                    int(config.image_size / config.vit_patches_size),
                )

            model = ViT_seg(
                vit_config,
                img_size=config.image_size,
                num_classes=vit_config.n_classes,
                vis=True,
            )

            pretrained_path = "./TransUNet/pretrained_weights/R50-ViT-B_16.npz"
            if os.path.exists(pretrained_path):
                try:
                    model.load_from(weights=np.load(pretrained_path))
                    print(f"✓ Loaded pretrained weights from {pretrained_path}")
                except Exception as e:
                    print(f"⚠️ Failed to load pretrained weights: {e}")
                    print("⚠️ Using random initialization!")
            else:
                print(f"⚠️ Pretrained weights not found at {pretrained_path}")
                print("⚠️ Using random initialization!")

            print(f"✓ Created TransUNet-{vit_name}")
            return model

        if architecture == "UnetPlusPlus":
            model = smp.UnetPlusPlus(encoder_name=config.backbone, encoder_weights="imagenet", activation=None)
            print(f"✓ Created {architecture} with {config.backbone}")
            return model

        if architecture == "FPN":
            model = smp.FPN(encoder_name=config.backbone, encoder_weights="imagenet", activation=None)
            print(f"✓ Created {architecture} with {config.backbone}")
            return model

        if architecture == "PSPNet":
            model = smp.PSPNet(encoder_name=config.backbone, encoder_weights="imagenet", activation=None)
            print(f"✓ Created {architecture} with {config.backbone}")
            return model

        if architecture == "DeepLabV3":
            model = smp.DeepLabV3(encoder_name=config.backbone, encoder_weights="imagenet", activation=None)
            print(f"✓ Created {architecture} with {config.backbone}")
            return model

        if architecture == "DeepLabV3Plus":
            model = smp.DeepLabV3Plus(encoder_name=config.backbone, encoder_weights="imagenet", activation=None)
            print(f"✓ Created {architecture} with {config.backbone}")
            return model

        raise ValueError(f"Unknown architecture: {architecture}")

    except Exception as e:
        print(f"❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        print("Falling back to resnet34 U-Net...")
        return smp.Unet("resnet34", encoder_weights="imagenet", activation=None)


# =========================
# 10) Experiment runner
# =========================
def run_experiment(config: Config):
    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {config.experiment_name}")
    print("=" * 80)
    print(f"Device: {config.device}")
    print(f"Fold: {config.fold}/{config.total_folds}")
    print(f"Backbone: {config.backbone}")
    print(f"Image size: {config.image_size}")
    print(f"Batch size: {config.batch_size}")
    print(f"Accumulation steps: {config.accumulation_steps}")
    print(f"Effective batch: {config.batch_size * config.accumulation_steps}")
    print(f"Epochs: {config.num_epochs}")

    print("\n--- Ablation Settings ---")
    print(f"Dynamic Sampling: {config.use_dynamic_sampling}")
    if config.use_dynamic_sampling:
        print(f" Initial positive ratio: {config.initial_positive_ratio}")
        print(f" Final positive ratio: {config.final_positive_ratio}")
    print(f"Heavy Augmentation: {config.use_heavy_augmentation}")
    print(f"Combo Loss: {config.use_combo_loss}")
    if config.use_combo_loss:
        print(f" Weights - BCE:{config.bce_weight}, Dice:{config.dice_weight}, Focal:{config.focal_weight}")
    print(f"Triplet Threshold: {config.use_triplet_threshold}")
    if config.use_triplet_threshold:
        print(f" Top:{config.triplet_top}, Bottom:{config.triplet_bottom}, Min area:{config.triplet_min_area}")
    print(f"Progressive Training: {config.use_progressive_training}")
    print(f"Progressive Resolution: {config.use_progressive_resolution}")
    if config.use_progressive_resolution:
        print(f" {config.initial_size} → {config.final_size}")

    print("=" * 80 + "\n")

    config.save()

    model = create_model(config)
    trainer = Trainer(model, config)
    trainer.start()

    print(f"\nBest validation loss: {trainer.best_loss:.4f}")

    def plot(scores, name):
        plt.figure(figsize=(15, 5))
        plt.plot(range(len(scores["train"])), scores["train"], label=f"train {name}")
        plt.plot(range(len(scores["train"])), scores["val"], label=f"val {name}")
        plt.title(f"{name} plot - {config.experiment_name}")
        plt.xlabel("Epoch")
        plt.ylabel(name)
        plt.legend()
        plt.savefig(config.save_dir / f"{name}_plot.png", dpi=150, bbox_inches="tight")
        plt.close()

    plot(trainer.losses, "Loss")
    plot(trainer.dice_scores, "Dice score")
    plot(trainer.iou_scores, "IoU score")

    summary = {
        "experiment_name": config.experiment_name,
        "fold": config.fold,
        "total_folds": config.total_folds,
        "num_epochs": config.num_epochs,
        "best_val_loss": float(trainer.best_loss),
        "final_train_loss": float(trainer.losses["train"][-1]),
        "final_val_loss": float(trainer.losses["val"][-1]),
        "final_train_dice": float(trainer.dice_scores["train"][-1]),
        "final_val_dice": float(trainer.dice_scores["val"][-1]),
        "final_train_iou": float(trainer.iou_scores["train"][-1]),
        "final_val_iou": float(trainer.iou_scores["val"][-1]),
        "use_dynamic_sampling": config.use_dynamic_sampling,
        "use_heavy_augmentation": config.use_heavy_augmentation,
        "use_combo_loss": config.use_combo_loss,
        "use_triplet_threshold": config.use_triplet_threshold,
        "use_progressive_training": config.use_progressive_training,
        "use_progressive_resolution": config.use_progressive_resolution,
        "backbone": config.backbone,
    }

    with open(config.save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("\n[FINAL RESULTS]")
    print(f"Best Val Loss: {trainer.best_loss:.4f}")
    print(f"Final Train Dice: {trainer.dice_scores['train'][-1]:.4f}")
    print(f"Final Val Dice: {trainer.dice_scores['val'][-1]:.4f}")
    print(f"Final Train IoU: {trainer.iou_scores['train'][-1]:.4f}")
    print(f"Final Val IoU: {trainer.iou_scores['val'][-1]:.4f}")

    return summary


# =========================
# 11) Helper: JSON serialize
# =========================
def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# =========================
# 12) Main
# =========================
def main():
    experiments = [
        Config(
            experiment_name="TransUnet_r34",
            model_architecture="TransUNet",
            backbone="resnet34",
            base_lr=5e-5,
            vit_patches_size=16,
            n_skip=3,
            use_dynamic_sampling=True,
            initial_positive_ratio=0.8,
            final_positive_ratio=0.4,
            use_heavy_augmentation=True,
            use_combo_loss=True,
            bce_weight=1.0,
            dice_weight=1.0,
            focal_weight=1.0,
            use_triplet_threshold=True,
            triplet_top=0.75,
            triplet_bottom=0.3,
            triplet_min_area=2000,
            image_size=512,
            batch_size=4,
            accumulation_steps=4,
            num_epochs=80,
            use_early_stopping=False,
            early_stopping_patience=10,
            early_stopping_metric="dice",
            early_stopping_min_delta=0.005,
        )
    ]

    all_results = []
    for i, config in enumerate(experiments):
        print("\n" + "#" * 80)
        print(f"RUNNING EXPERIMENT {i + 1}/{len(experiments)}")
        print("#" * 80 + "\n")

        try:
            result = run_experiment(config)
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ Error in {config.experiment_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        print(f"\n✓ Completed {config.experiment_name}")
        print("=" * 80 + "\n")

    # 여기에 “노트북 뒷부분의 테스트 평가 코드”까지 합치고 싶으면,
    # 기존에 올린 test/eval 루틴을 별도 파일(test_saa.py 등)로 분리하는 걸 추천.
    # (이 파일은 학습 파이프라인만 깔끔히 유지)

    return all_results


if __name__ == "__main__":
    main()
