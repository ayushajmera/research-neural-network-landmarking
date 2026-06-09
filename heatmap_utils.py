"""
heatmap_utils.py
----------------
Utility functions for converting insect wing vein junction coordinates
to heatmaps and converting predicted heatmaps back to junction points.

Used by:
    train_detector.py
    inference.py
    app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter, label, center_of_mass


def make_gaussian_kernel(sigma: int = 8) -> np.ndarray:
    """
    Create a 2D Gaussian kernel.

    Kernel size is:
        sigma * 6 + 1

    This covers roughly +/- 3 sigma around the centre.
    """

    sigma = int(sigma)
    kernel_size = sigma * 6 + 1

    ax = np.arange(kernel_size) - kernel_size // 2
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.max()

    return kernel.astype(np.float32)


def points_to_heatmap(
    points: Iterable[Tuple[float, float]],
    image_h: int,
    image_w: int,
    sigma: int = 8,
) -> np.ndarray:
    """
    Convert a list of junction coordinates into a heatmap.

    Parameters
    ----------
    points:
        List of (x, y) coordinates.

    image_h:
        Height of output heatmap.

    image_w:
        Width of output heatmap.

    sigma:
        Controls the size of each Gaussian blob.

    Returns
    -------
    heatmap:
        Float32 numpy array of shape (image_h, image_w), range [0, 1].
    """

    heatmap = np.zeros((image_h, image_w), dtype=np.float32)
    kernel = make_gaussian_kernel(sigma)

    kernel_h, kernel_w = kernel.shape
    radius_y = kernel_h // 2
    radius_x = kernel_w // 2

    for x, y in points:
        x = int(round(float(x)))
        y = int(round(float(y)))

        # Skip points outside image
        if x < 0 or x >= image_w or y < 0 or y >= image_h:
            continue

        # Region in heatmap
        x1 = max(0, x - radius_x)
        x2 = min(image_w, x + radius_x + 1)
        y1 = max(0, y - radius_y)
        y2 = min(image_h, y + radius_y + 1)

        # Matching region in Gaussian kernel
        kx1 = radius_x - (x - x1)
        kx2 = kx1 + (x2 - x1)
        ky1 = radius_y - (y - y1)
        ky2 = ky1 + (y2 - y1)

        # Use maximum so overlapping junction blobs do not cancel each other
        heatmap[y1:y2, x1:x2] = np.maximum(
            heatmap[y1:y2, x1:x2],
            kernel[ky1:ky2, kx1:kx2],
        )

    heatmap = np.clip(heatmap, 0.0, 1.0)

    return heatmap.astype(np.float32)


def csv_to_heatmap(
    csv_path: str | Path,
    image_h: int,
    image_w: int,
    sigma: int = 8,
) -> np.ndarray:
    """
    Read a CSV file with x,y columns and convert coordinates to heatmap.

    CSV format:
        x,y
        120,340
        188,292
        260,410
    """

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if not {"x", "y"}.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain columns named 'x' and 'y'.")

    points = list(zip(df["x"].astype(float), df["y"].astype(float)))

    return points_to_heatmap(
        points=points,
        image_h=image_h,
        image_w=image_w,
        sigma=sigma,
    )


def heatmap_to_points(
    heatmap: np.ndarray,
    threshold: float = 0.4,
    min_distance: int = 10,
) -> List[Tuple[int, int]]:
    """
    Convert predicted heatmap into junction coordinates.

    Steps:
    1. Smoothly find local maxima using maximum_filter.
    2. Keep only pixels above threshold.
    3. Label connected peak regions.
    4. Return weighted centroid of each region.

    Parameters
    ----------
    heatmap:
        2D heatmap array.

    threshold:
        Minimum heatmap confidence required.

    min_distance:
        Minimum spacing between detected junction clusters.

    Returns
    -------
    points:
        List of integer (x, y) coordinate tuples.
    """

    if heatmap.ndim == 3:
        heatmap = np.squeeze(heatmap)

    if heatmap.ndim != 2:
        raise ValueError(
            f"heatmap_to_points expects a 2D heatmap, got shape {heatmap.shape}"
        )

    heatmap = heatmap.astype(np.float32)

    # Remove tiny numerical noise
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=1.0, neginf=0.0)

    # Local maximum detection
    footprint_size = max(1, int(min_distance))

    local_max = heatmap == maximum_filter(
        heatmap,
        size=footprint_size,
        mode="constant",
    )

    # Keep only confident local maxima
    peak_mask = local_max & (heatmap >= threshold)

    # Connected components of neighbouring peak pixels
    labelled, num_features = label(peak_mask)

    points: List[Tuple[int, int]] = []

    for component_id in range(1, num_features + 1):
        component_mask = labelled == component_id

        if not np.any(component_mask):
            continue

        # Weighted centroid gives more stable centre than just argmax
        weights = heatmap * component_mask

        total_weight = weights.sum()

        if total_weight <= 0:
            ys, xs = np.where(component_mask)
            cx = int(round(xs.mean()))
            cy = int(round(ys.mean()))
        else:
            cy, cx = center_of_mass(weights)
            cx = int(round(cx))
            cy = int(round(cy))

        points.append((cx, cy))

    # Sort top-to-bottom, then left-to-right for consistent output
    points = sorted(points, key=lambda p: (p[1], p[0]))

    return points


if __name__ == "__main__":
    # Small self-test
    test_points = [(50, 50), (100, 120), (200, 80)]

    hm = points_to_heatmap(
        points=test_points,
        image_h=256,
        image_w=256,
        sigma=8,
    )

    detected = heatmap_to_points(
        hm,
        threshold=0.4,
        min_distance=10,
    )

    print("Original points:", test_points)
    print("Detected points:", detected)
    print("Heatmap shape:", hm.shape)
    print("Heatmap min/max:", hm.min(), hm.max())