"""
fai_heatmap_visualizer.py (public-safe)

Generates false-color heatmaps of Floating Algae Index (FAI) intensity
for each *_FAI.tif file in a user-supplied directory.

- Displays a grid overview
- Optionally saves each heatmap individually

NOTE: Raw Sentinel-2/FAI data is NOT included in this repository.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio


# ============================================================
# Helper functions
# ============================================================

def extract_date(path: Path) -> str:
    m = re.search(r"(\d{4}_\d{2}_\d{2})", path.name)
    return m.group(1) if m else path.stem


def load_fai(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        fai = src.read(1).astype(np.float32)
    return fai


# ============================================================
# Grid visualization + individual saves
# ============================================================

def grid_view(
    fai_files: list[Path],
    cols: int = 4,
    thumb_size: int = 4,
    save_dir: Path | None = None,
):
    n = len(fai_files)
    if n == 0:
        print("No FAI TIFFs found.")
        return

    rows = math.ceil(n / cols)
    plt.figure(figsize=(cols * thumb_size, rows * thumb_size))

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    for idx, f in enumerate(fai_files):
        fai = load_fai(f)
        date = extract_date(f)

        # --- Grid subplot ---
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(fai, cmap="inferno")
        plt.title(f"FAI {date}", fontsize=10)
        plt.axis("off")

        # --- Save individual image (optional) ---
        if save_dir is not None:
            save_path = save_dir / f"fai_heatmap_{date}.png"

            fig = plt.figure(figsize=(6, 5))
            plt.imshow(fai, cmap="inferno")
            plt.colorbar(label="FAI Intensity")
            plt.title(f"FAI Heatmap - {date}")
            plt.axis("off")
            plt.tight_layout()
            fig.savefig(save_path, dpi=150)
            plt.close(fig)

            print("Saved:", save_path)

    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Folder containing *_FAI.tif files (not included in repo).",
    )
    ap.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional folder to save individual heatmaps (PNG).",
    )
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--thumb-size", type=int, default=4)
    args = ap.parse_args()

    if not args.raw_dir.exists():
        raise RuntimeError("Input directory not found. Raw data is not included in this repository.")

    fai_files = sorted(list(args.raw_dir.glob("*_FAI.tif")) + list(args.raw_dir.glob("*_FAI.tiff")))
    if not fai_files:
        print("No FAI files found.")
        return

    print(f"Found {len(fai_files)} FAI files.")
    grid_view(fai_files, cols=args.cols, thumb_size=args.thumb_size, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
