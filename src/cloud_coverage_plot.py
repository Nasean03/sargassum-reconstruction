"""
cloud_coverage_plot.py (public-safe)

Generates:
1) A bar chart (cloud coverage per scene)
2) A histogram (distribution across scenes)

Cloud coverage is estimated using RGB-only cloud/haze masking,
then averaged at tile level (supports partial edge tiles).

NOTE: Raw data is NOT included in this repository.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import rasterio


# ============================================================
# Helper functions
# ============================================================

def extract_date(path: Path) -> str:
    m = re.search(r"(\d{4}_\d{2}_\d{2})", path.name)
    return m.group(1) if m else path.stem


def load_rgb_as_float(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        rgb = src.read([1, 2, 3]).astype(np.float32)
    rgb = np.transpose(rgb, (1, 2, 0))
    return np.clip(rgb / 10000.0, 0, 1)


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


def compute_scene_cloud_coverage(mask: np.ndarray, tile_size: int) -> float:
    """
    Computes mean cloud fraction across tiles (0..1),
    using ceil-like tiling so edge tiles are included.
    """
    H, W = mask.shape
    rows = (H + tile_size - 1) // tile_size
    cols = (W + tile_size - 1) // tile_size

    vals = []
    for i in range(rows):
        for j in range(cols):
            y0 = i * tile_size
            y1 = min((i + 1) * tile_size, H)
            x0 = j * tile_size
            x1 = min((j + 1) * tile_size, W)
            t = mask[y0:y1, x0:x1]
            vals.append(float(t.mean()) if t.size else 0.0)

    return float(np.mean(vals)) if vals else 0.0


def find_rgb_files(raw_dir: Path) -> list[Path]:
    return sorted(list(raw_dir.glob("*_RGB.tif")) + list(raw_dir.glob("*_RGB.tiff")))


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, required=True, help="Folder containing *_RGB.tif(f) files (not included).")
    ap.add_argument("--tile-size", type=int, default=16, help="Tile size in pixels (default: 16).")
    ap.add_argument("--save-dir", type=Path, default=None, help="Optional folder to save plots as PNG.")
    args = ap.parse_args()

    if not args.raw_dir.exists():
        raise RuntimeError("Input directory not found. Raw data is not included in this repository.")

    rgb_files = find_rgb_files(args.raw_dir)
    if not rgb_files:
        print("No RGB files found.")
        return

    cloud_fracs = []
    labels = []

    for f in rgb_files:
        print("Processing:", f.name)
        rgb = load_rgb_as_float(f)
        mask = compute_cloud_mask_from_rgb(rgb)
        frac = compute_scene_cloud_coverage(mask, args.tile_size)
        cloud_fracs.append(frac)
        labels.append(extract_date(f))

    cloud_fracs = np.array(cloud_fracs) * 100.0  # %
    n = len(cloud_fracs)

    # --- Bar Chart ---
    plt.figure(figsize=(10, 5))
    plt.bar(range(n), cloud_fracs, edgecolor="black")
    plt.xticks(range(n), labels, rotation=45, ha="right")
    plt.ylabel("Cloud Coverage (%)")
    plt.title("Cloud Coverage per Scene")
    plt.tight_layout()

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        out1 = args.save_dir / "cloud_coverage_per_scene.png"
        plt.savefig(out1, dpi=200)
        print("Saved:", out1)

    plt.show()

    # --- Histogram ---
    plt.figure(figsize=(8, 5))
    plt.hist(cloud_fracs, bins=10, edgecolor="black")
    plt.xlabel("Cloud Coverage (%)")
    plt.ylabel("Number of Scenes")
    plt.title("Distribution of Cloud Coverage Across Scenes")
    plt.tight_layout()

    if args.save_dir is not None:
        out2 = args.save_dir / "cloud_coverage_distribution_hist.png"
        plt.savefig(out2, dpi=200)
        print("Saved:", out2)

    plt.show()


if __name__ == "__main__":
    main()
