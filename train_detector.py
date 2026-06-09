"""
train_detector.py
-----------------
Train a U-Net heatmap-regression model to detect insect wing vein junctions.

Expected data layout:
    data/images/wing_001.png
    data/landmarks/wing_001.csv

Each CSV must contain columns:
    x, y
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from scipy.spatial import cKDTree
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset

from heatmap_utils import heatmap_to_points, points_to_heatmap


# ============================================================================
# CONFIG - edit these paths/settings for your own project
# ============================================================================
IMAGE_DIR = Path("data/images")
LANDMARK_DIR = Path("data/landmarks")
IMAGE_SIZE = 512
SIGMA = 8
BATCH_SIZE = 4
EPOCHS = 80
LR = 1e-4
VAL_SPLIT = 0.15
SAVE_PATH = Path("best_junction_detector.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def read_image_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_and_scale_points(
    csv_path: Path,
    old_w: int,
    old_h: int,
    new_w: int,
    new_h: int,
) -> List[Tuple[float, float]]:
    df = pd.read_csv(csv_path)

    if not {"x", "y"}.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain x and y columns")

    scale_x = new_w / float(old_w)
    scale_y = new_h / float(old_h)

    points = []
    for x, y in zip(df["x"].astype(float), df["y"].astype(float)):
        sx = float(x) * scale_x
        sy = float(y) * scale_y

        if 0 <= sx < new_w and 0 <= sy < new_h:
            points.append((sx, sy))

    return points


class WingJunctionDataset(Dataset):
    """
    Dataset that pairs every image with the CSV file sharing the same stem.

    Important:
    Heatmaps are rendered AFTER Albumentations transforms the keypoints.
    This keeps the augmented image and heatmap label aligned.
    """

    def __init__(
        self,
        image_dir: Path,
        landmark_dir: Path,
        image_size: int = IMAGE_SIZE,
        sigma: int = SIGMA,
        transform: A.Compose | None = None,
    ):
        self.image_dir = Path(image_dir)
        self.landmark_dir = Path(landmark_dir)
        self.image_size = int(image_size)
        self.sigma = sigma
        self.transform = transform
        self.samples = self._find_pairs()

        if not self.samples:
            raise RuntimeError(
                f"No image/CSV pairs found. Checked images in {self.image_dir} "
                f"and CSVs in {self.landmark_dir}."
            )

    def _find_pairs(self) -> List[Tuple[Path, Path]]:
        image_paths = sorted(
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )

        pairs = []
        missing = []

        for img_path in image_paths:
            csv_path = self.landmark_dir / f"{img_path.stem}.csv"

            if csv_path.exists():
                pairs.append((img_path, csv_path))
            else:
                missing.append(img_path.name)

        if missing:
            print(f"Warning: {len(missing)} image(s) skipped because matching CSV was missing.")

        return pairs

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, csv_path = self.samples[idx]

        image = read_image_rgb(img_path)
        old_h, old_w = image.shape[:2]

        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )

        keypoints = load_and_scale_points(
            csv_path,
            old_w,
            old_h,
            self.image_size,
            self.image_size,
        )

        if self.transform is not None:
            transformed = self.transform(image=image, keypoints=keypoints)
            image_tensor = transformed["image"]
            keypoints_aug = transformed["keypoints"]
        else:
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            keypoints_aug = keypoints

        valid_keypoints = [
            (float(x), float(y))
            for x, y in keypoints_aug
            if 0 <= float(x) < self.image_size and 0 <= float(y) < self.image_size
        ]

        heatmap = points_to_heatmap(
            valid_keypoints,
            self.image_size,
            self.image_size,
            sigma=self.sigma,
        )

        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).float()

        return {
            "image": image_tensor.float(),
            "heatmap": heatmap_tensor,
            "name": img_path.stem,
        }


def build_train_transform() -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.03,
                scale_limit=0.05,
                rotate_limit=10,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.4,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
            A.GaussNoise(var_limit=(5, 30), p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(
            format="xy",
            remove_invisible=True,
        ),
    )


def build_val_transform() -> A.Compose:
    return A.Compose(
        [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(
            format="xy",
            remove_invisible=True,
        ),
    )


def build_model(encoder_weights: str | None = "imagenet") -> nn.Module:
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation="sigmoid",
    )


def weighted_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 20.0,
) -> torch.Tensor:
    weights = 1.0 + (pos_weight - 1.0) * target
    return torch.mean(weights * (pred - target) ** 2)


def match_points_f1(
    pred_points: Sequence[Tuple[int, int]],
    gt_points: Sequence[Tuple[int, int]],
    tolerance_px: float = 10.0,
) -> Tuple[float, float, float]:

    if len(pred_points) == 0 and len(gt_points) == 0:
        return 1.0, 1.0, 1.0

    if len(pred_points) == 0:
        return 0.0, 0.0, 0.0

    if len(gt_points) == 0:
        return 0.0, 0.0, 0.0

    pred = np.asarray(pred_points, dtype=np.float32)
    gt = np.asarray(gt_points, dtype=np.float32)

    tree = cKDTree(gt)

    matched_gt = set()
    true_pos = 0

    for p in pred:
        distances, indices = tree.query(
            p,
            k=min(len(gt), 5),
            distance_upper_bound=tolerance_px,
        )

        distances = np.atleast_1d(distances)
        indices = np.atleast_1d(indices)

        best_idx = None
        best_dist = float("inf")

        for dist, idx in zip(distances, indices):
            idx = int(idx)

            if (
                np.isfinite(dist)
                and idx < len(gt)
                and idx not in matched_gt
                and dist < best_dist
            ):
                best_idx = idx
                best_dist = float(dist)

        if best_idx is not None:
            matched_gt.add(best_idx)
            true_pos += 1

    precision = true_pos / max(len(pred_points), 1)
    recall = true_pos / max(len(gt_points), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return precision, recall, f1


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    threshold: float = 0.4,
    min_distance: int = 10,
    tolerance_px: int = 10,
) -> Tuple[float, float]:

    model.eval()

    total_loss = 0.0
    total_f1 = 0.0
    n_batches = 0
    n_images = 0

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["heatmap"].to(device)

        preds = model(images)

        loss = weighted_mse_loss(
            preds,
            targets,
            pos_weight=10.0,
        )

        total_loss += float(loss.item())
        n_batches += 1

        preds_np = preds.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        for pred_hm, gt_hm in zip(preds_np, targets_np):
            pred_points = heatmap_to_points(
                pred_hm[0],
                threshold=threshold,
                min_distance=min_distance,
            )

            gt_points = heatmap_to_points(
                gt_hm[0],
                threshold=0.25,
                min_distance=min_distance,
            )

            _, _, f1 = match_points_f1(
                pred_points,
                gt_points,
                tolerance_px=tolerance_px,
            )

            total_f1 += f1
            n_images += 1

    return (
        total_loss / max(n_batches, 1),
        total_f1 / max(n_images, 1),
    )


def split_indices(
    n: int,
    val_split: float,
    seed: int,
) -> Tuple[List[int], List[int]]:

    indices = list(range(n))

    rng = random.Random(seed)
    rng.shuffle(indices)

    val_count = max(1, int(round(n * val_split))) if n > 1 else 0

    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    return train_indices, val_indices


def main() -> None:
    set_seed(SEED)

    print(f"Using device: {DEVICE}")

    train_dataset_full = WingJunctionDataset(
        IMAGE_DIR,
        LANDMARK_DIR,
        IMAGE_SIZE,
        SIGMA,
        transform=build_train_transform(),
    )

    val_dataset_full = WingJunctionDataset(
        IMAGE_DIR,
        LANDMARK_DIR,
        IMAGE_SIZE,
        SIGMA,
        transform=build_val_transform(),
    )

    train_indices, val_indices = split_indices(
        len(train_dataset_full),
        VAL_SPLIT,
        SEED,
    )

    if not train_indices or not val_indices:
        raise RuntimeError(
            "Need at least 2 labelled images for a train/validation split."
        )

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = build_model(encoder_weights="imagenet").to(DEVICE)

    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    best_f1 = -1.0
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            images = batch["image"].to(DEVICE)
            targets = batch["heatmap"].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            preds = model(images)

            loss = weighted_mse_loss(
                preds,
                targets,
                pos_weight=10.0,
            )

            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            n_batches += 1

        scheduler.step()

        train_loss /= max(n_batches, 1)

        val_loss, val_f1 = validate(
            model,
            val_loader,
            DEVICE,
            threshold=0.4,
            min_distance=10,
            tolerance_px=10,
        )

        print(
            f"Epoch {epoch}/{EPOCHS}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_F1={val_f1:.3f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "image_size": IMAGE_SIZE,
                "sigma": SIGMA,
                "best_f1": best_f1,
                "epoch": epoch,
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
            }

            torch.save(checkpoint, SAVE_PATH)

            print(
                f"Saved new best model to {SAVE_PATH} "
                f"with val_F1={best_f1:.3f}"
            )

    print(
        f"Training complete. "
        f"Best val_F1={best_f1:.3f}. "
        f"Best weights: {SAVE_PATH}"
    )


if __name__ == "__main__":
    main()