"""
validation.py (public-safe)

Validates a reconstructed image by showing:
1) Reconstructed RGB
2) Cloud mask visualization
3) FAI heatmap (if bands available)

All stacked vertically in one figure.

NOTE: Raw/reconstructed data is NOT included in this repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import rasterio


# ============================================================
# Helper functions
# ============================================================

def compute_cloud_mask_from_rgb(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    brightness = (r + g + b) / 3
    maxc = np.max(rgb, axis=2)
    minc = np.min(rgb, axis=2)
    sat = (maxc - minc) / (brightness + 1e-6)

    bright_cloud = (brightness > 0.40) & (sat < 0.35)
    haze_cloud = (
        (0.15 < brightness) & (brightness < 0.45)
        & (sat < 0.22) & (b > r) & (b > g)
    )
    return (bright_cloud | haze_cloud).astype(np.uint8)


def compute_fai(nir: np.ndarray, red: np.ndarray, swir: np.ndarray) -> np.ndarray:
    # Floating Algae Index (FAI)
    return nir - (red + (swir - red) * (0.86 - 0.66) / (1.61 - 0.66))


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tif",
        type=Path,
        required=True,
        help="Path to reconstructed GeoTIFF (not included in repo).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to save the validation PNG (e.g., outputs/validation.png).",
    )
    args = ap.parse_args()

    if not args.tif.exists():
        raise RuntimeError("Reconstructed TIFF not found. Provide a valid path.")

    with rasterio.open(args.tif) as src:
        arr = src.read().astype(np.float32)  # (bands, H, W)

    if arr.shape[0] < 3:
        raise RuntimeError("Expected at least 3 bands (R,G,B).")

    # RGB assumes first three bands are R,G,B (reflectance scaled by 10000)
    rgb = np.transpose(arr[0:3], (1, 2, 0))
    rgb = np.clip(rgb / 10000.0, 0, 1)

    # Cloud mask
    mask = compute_cloud_mask_from_rgb(rgb)

    # FAI (if NIR and SWIR bands exist)
    if arr.shape[0] >= 5:
        red = arr[0] / 10000.0
        nir = arr[3] / 10000.0
        swir = arr[4] / 10000.0
        fai = compute_fai(nir, red, swir)
        fai_title = "FAI Heatmap (Reconstructed)"
    else:
        # Proxy: show red reflectance if NIR/SWIR not available
        print("No NIR/SWIR bands found. Using red reflectance as proxy heatmap.")
        fai = arr[0] / 10000.0
        fai_title = "Proxy Heatmap (Red Reflectance)"

    # --- Plot stacked figure ---
    fig = plt.figure(figsize=(8, 15))

    plt.subplot(3, 1, 1)
    plt.imshow(rgb)
    plt.title("Reconstructed RGB")
    plt.axis("off")

    plt.subplot(3, 1, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Cloud Mask (Reconstructed)")
    plt.axis("off")

    plt.subplot(3, 1, 3)
    plt.imshow(fai, cmap="inferno")
    plt.colorbar(label="FAI Intensity")
    plt.title(fai_title)
    plt.axis("off")

    plt.tight_layout()

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=200)
        print("Saved validation figure to:", args.out)

    plt.show()


if __name__ == "__main__":
    main()
