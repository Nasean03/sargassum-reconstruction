"""
cloud_mask_visualizer.py (public-safe)

Generates a Figure-style comparison:
Left  = Original RGB
Right = Cloud Mask (white = cloud/haze, black = clear)

NOTE: Raw Sentinel-2 data is NOT included in this repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt


# ================================================
# LOAD RGB
# ================================================
def load_rgb_as_float(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        rgb = src.read([1, 2, 3]).astype(np.float32)
    rgb = np.transpose(rgb, (1, 2, 0))
    return np.clip(rgb / 10000.0, 0, 1)


# ================================================
# CLOUD MASK (same logic as main script)
# ================================================
def compute_cloud_mask_from_rgb(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    brightness = (r + g + b) / 3

    maxc = np.max(rgb, axis=2)
    minc = np.min(rgb, axis=2)
    sat = (maxc - minc) / (brightness + 1e-6)

    bright_cloud = (brightness > 0.40) & (sat < 0.35)

    haze_cloud = (
        (0.15 < brightness) & (brightness < 0.45)
        & (sat < 0.22)
        & (b > r) & (b > g)
    )

    return (bright_cloud | haze_cloud).astype(np.uint8)


# ================================================
# VISUALIZATION
# ================================================
def show_cloud_mask(rgb: np.ndarray, mask: np.ndarray, out_prefix: Path | None = None):
    plt.figure(figsize=(12, 6))

    # LEFT: Original RGB
    plt.subplot(1, 2, 1)
    plt.imshow(rgb)
    plt.title("Original RGB Image")
    plt.axis("off")

    # RIGHT: Cloud Mask
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Cloud Mask (white = cloud/haze)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Optional saves
    if out_prefix is not None:
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        rgb_path = out_prefix.with_suffix("").as_posix() + "_orig.png"
        mask_path = out_prefix.with_suffix("").as_posix() + "_mask.png"

        plt.imsave(rgb_path, rgb)
        plt.imsave(mask_path, mask, cmap="gray")
        print(f"Saved: {rgb_path}")
        print(f"Saved: {mask_path}")


# ================================================
# MAIN
# ================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", type=Path, required=True, help="Path to a *_RGB.tif file (not included in repo).")
    ap.add_argument(
        "--out-prefix",
        type=Path,
        default=None,
        help="Optional output prefix, e.g. outputs/figure2_cloud (saves *_orig.png and *_mask.png)",
    )
    args = ap.parse_args()

    if not args.rgb.exists():
        raise RuntimeError("RGB file not found. Raw data is not included in this repository.")

    print("Loading RGB image...")
    rgb = load_rgb_as_float(args.rgb)

    print("Computing cloud mask...")
    mask = compute_cloud_mask_from_rgb(rgb)

    print("Displaying figure...")
    show_cloud_mask(rgb, mask, out_prefix=args.out_prefix)


if __name__ == "__main__":
    main()
