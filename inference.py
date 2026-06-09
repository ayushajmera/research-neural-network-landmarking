"""
inference.py
------------
Run trained wing junction detector on new insect wing images.

Input:
    clean wing image

Output:
    - detected junction coordinates
    - heatmap
    - overlay image
    - CSV file if used from command line
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as smp

from heatmap_utils import heatmap_to_points


# Must match training
IMAGE_SIZE = 512
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_model() -> torch.nn.Module:
    """
    Build the same U-Net architecture used during training.

    During inference we use encoder_weights=None because weights
    are loaded from our trained checkpoint.
    """

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation="sigmoid",
    )

    return model


def load_detector(
    weights_path: str | Path,
    device: str | None = None,
) -> torch.nn.Module:
    """
    Load trained junction detector.

    Supports both checkpoint formats:
    1. torch.save(model.state_dict(), path)
    2. torch.save({"model_state_dict": model.state_dict(), ...}, path)
    """

    weights_path = Path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model()

    checkpoint = torch.load(
        weights_path,
        map_location=device,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def preprocess_image(image_bgr: np.ndarray) -> torch.Tensor:
    """
    Resize and normalize image exactly like validation/inference training setup.
    """

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(
        image_rgb,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    resized = resized.astype(np.float32) / 255.0
    resized = (resized - IMAGENET_MEAN) / IMAGENET_STD

    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float()
    tensor = tensor.unsqueeze(0)

    return tensor


@torch.no_grad()
def detect_junctions(
    image_bgr: np.ndarray,
    model: torch.nn.Module,
    threshold: float = 0.30,
    min_distance: int = 10,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Detect junction points from a clean wing image.

    Returns
    -------
    junctions:
        List of (x, y) coordinates in ORIGINAL image size.

    heatmap_fullres:
        Predicted heatmap resized back to original image size.
    """

    if image_bgr is None:
        raise ValueError("image_bgr is None")

    original_h, original_w = image_bgr.shape[:2]

    device = next(model.parameters()).device

    input_tensor = preprocess_image(image_bgr).to(device)

    pred = model(input_tensor)

    heatmap = pred.squeeze().detach().cpu().numpy().astype(np.float32)

    # Detect points at model resolution
    points_model = heatmap_to_points(
        heatmap,
        threshold=threshold,
        min_distance=min_distance,
    )

    # Scale points back to original image resolution
    scale_x = original_w / float(IMAGE_SIZE)
    scale_y = original_h / float(IMAGE_SIZE)

    junctions = []

    for x, y in points_model:
        ox = int(round(x * scale_x))
        oy = int(round(y * scale_y))

        ox = max(0, min(original_w - 1, ox))
        oy = max(0, min(original_h - 1, oy))

        junctions.append((ox, oy))

    # Resize heatmap to original image size for display
    heatmap_fullres = cv2.resize(
        heatmap,
        (original_w, original_h),
        interpolation=cv2.INTER_LINEAR,
    )

    heatmap_fullres = np.clip(heatmap_fullres, 0.0, 1.0)

    return junctions, heatmap_fullres


def create_overlay(
    image_bgr: np.ndarray,
    junctions: List[Tuple[int, int]],
    heatmap: np.ndarray | None = None,
    alpha: float = 0.35,
) -> np.ndarray:
    """
    Create overlay image with optional heatmap and junction dots.

    Drawing style:
    - optional JET heatmap
    - white outer circle
    - dark green ring
    - green filled centre
    """

    overlay = image_bgr.copy()

    if heatmap is not None:
        hm = np.clip(heatmap, 0.0, 1.0)
        hm_uint8 = (hm * 255).astype(np.uint8)

        heatmap_color = cv2.applyColorMap(
            hm_uint8,
            cv2.COLORMAP_JET,
        )

        overlay = cv2.addWeighted(
            overlay,
            1.0 - alpha,
            heatmap_color,
            alpha,
            0,
        )

    for x, y in junctions:
        # white outer ring
        cv2.circle(
            overlay,
            (int(x), int(y)),
            8,
            (255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        # dark green ring
        cv2.circle(
            overlay,
            (int(x), int(y)),
            6,
            (0, 90, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        # filled bright green centre
        cv2.circle(
            overlay,
            (int(x), int(y)),
            4,
            (0, 255, 0),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    return overlay


def save_outputs(
    image_path: Path,
    junctions: List[Tuple[int, int]],
    overlay_bgr: np.ndarray,
) -> Tuple[Path, Path]:
    """
    Save detected coordinates and overlay image.
    """

    out_csv = image_path.with_name(f"{image_path.stem}_junctions.csv")
    out_img = image_path.with_name(f"{image_path.stem}_junctions.png")

    df = pd.DataFrame(
        junctions,
        columns=["x", "y"],
    )

    df.to_csv(out_csv, index=False)
    cv2.imwrite(str(out_img), overlay_bgr)

    return out_csv, out_img


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image",
        type=str,
        help="Path to input wing image",
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="best_junction_detector.pth",
        help="Path to trained model weights",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="Detection threshold. Try 0.25-0.35 for weak models.",
    )

    parser.add_argument(
        "--min-distance",
        type=int,
        default=10,
        help="Minimum spacing between junction detections",
    )

    parser.add_argument(
        "--show-heatmap",
        action="store_true",
        help="Blend heatmap into output overlay",
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    model = load_detector(args.weights)

    junctions, heatmap = detect_junctions(
        image_bgr=image_bgr,
        model=model,
        threshold=args.threshold,
        min_distance=args.min_distance,
    )

    overlay = create_overlay(
        image_bgr=image_bgr,
        junctions=junctions,
        heatmap=heatmap if args.show_heatmap else None,
    )

    out_csv, out_img = save_outputs(
        image_path=image_path,
        junctions=junctions,
        overlay_bgr=overlay,
    )

    print(f"Detected junctions: {len(junctions)}")
    print(f"Saved CSV: {out_csv}")
    print(f"Saved overlay: {out_img}")


if __name__ == "__main__":
    main()