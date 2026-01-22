"""
tile_grid_visualizer.py (public-safe)

Overlays a tile grid on a Sentinel-2 RGB image (contrast-enhanced for visibility).

Modes:
- Provide --rgb to draw grid on a specific RGB GeoTIFF
- Provide --raw-dir to auto-select a "reference" image (cloudiest by RGB mask)

NOTE: Raw Sentinel-2 data is NOT included in this repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import rasterio
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


PIXEL_SIZE_M = 10  # Sentinel-2 RGB ~10 m / pixel


# ============================================================
# UTIL
# ============================================================
def extract_date(path: Path) -> str | None:
    m = re.search(r"(\d{4}_\d{2}_\d{2})", path.name)
    return m.group(1) if m else None


# ============================================================
# CONTRAST ENHANCEMENT
# ============================================================
def enhance_contrast(rgb: np.ndarray) -> np.ndarray:
    """Auto-stretch using 2–98% percentiles (display only)."""
    p2 = np.percentile(rgb, 2)
    p98 = np.percentile(rgb, 98)
    stretched = (rgb - p2) / (p98 - p2 + 1e-6)
    return np.clip(stretched, 0, 1)


# ============================================================
# LOADING
# ============================================================
def load_rgb_as_float(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        rgb = src.read([1, 2, 3]).astype(np.float32)
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = np.clip(rgb / 10000.0, 0, 1)
    return enhance_contrast(rgb)


# ============================================================
# CLOUD MASK (same heuristic)
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
        & (sat < 0.22)
        & (b > r) & (b > g)
    )
    return (bright_cloud | haze_cloud).astype(np.uint8)


def compute_tile_cloud_fraction(mask: np.ndarray, tile_size: int) -> float:
    """Returns mean cloud fraction across all tiles (supports edge tiles)."""
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
            tile = mask[y0:y1, x0:x1]
            vals.append(float(tile.mean()) if tile.size else 0.0)

    return float(np.mean(vals)) if vals else 0.0


# ============================================================
# FIND RGB FILES (RGB-only, no QA60 requirement)
# ============================================================
def find_rgb_files(raw_dir: Path) -> list[Path]:
    return sorted(list(raw_dir.glob("*_RGB.tif")) + list(raw_dir.glob("*_RGB.tiff")))


# ============================================================
# DRAW TILE GRID
# ============================================================
def draw_tile_grid(rgb: np.ndarray, tile_size: int, title: str = ""):
    img_uint8 = (rgb * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    draw = ImageDraw.Draw(pil_img)

    h, w, _ = rgb.shape

    x_edges = list(range(0, w, tile_size)) + [w]
    y_edges = list(range(0, h, tile_size)) + [h]

    for x in x_edges:
        draw.line([(x, 0), (x, h)], fill=(255, 0, 0), width=1)
    for y in y_edges:
        draw.line([(0, y), (w, y)], fill=(255, 0, 0), width=1)

    rows = (h + tile_size - 1) // tile_size
    cols = (w + tile_size - 1) // tile_size
    total_tiles = rows * cols

    meters_per_tile = tile_size * PIXEL_SIZE_M

    plt.figure(figsize=(10, 6))
    plt.imshow(pil_img)
    plt.title(
        f"{title}\n"
        f"Tile size: {tile_size} px  ({meters_per_tile} m × {meters_per_tile} m)\n"
        f"Total tiles: {total_tiles}",
        fontsize=15,
        weight="bold",
    )
    plt.axis("off")
    plt.show()


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile-size", type=int, default=16, help="Tile size in pixels.")
    ap.add_argument("--rgb", type=Path, default=None, help="Path to a specific *_RGB.tif(f).")
    ap.add_argument("--raw-dir", type=Path, default=None, help="Folder containing *_RGB.tif(f) files.")
    args = ap.parse_args()

    if args.rgb is None and args.raw_dir is None:
        raise RuntimeError("Provide either --rgb <file> OR --raw-dir <folder>.")

    # Mode A: user supplies a specific file
    if args.rgb is not None:
        if not args.rgb.exists():
            raise RuntimeError("RGB file not found. Raw data is not included in this repository.")
        rgb = load_rgb_as_float(args.rgb)
        draw_tile_grid(rgb, args.tile_size, title=f"Tile Grid Overlay\n{args.rgb.name}")
        return

    # Mode B: auto-pick a reference from a folder (cloudiest)
    if not args.raw_dir.exists():
        raise RuntimeError("Input directory not found. Raw data is not included in this repository.")

    rgb_files = find_rgb_files(args.raw_dir)
    if not rgb_files:
        print("No RGB files found in:", args.raw_dir)
        return

    rgb_stack = [load_rgb_as_float(p) for p in rgb_files]

    cloud_fracs = []
    for rgb in rgb_stack:
        mask = compute_cloud_mask_from_rgb(rgb)
        cloud_fracs.append(compute_tile_cloud_fraction(mask, args.tile_size))

    ref_idx = int(np.argmax(cloud_fracs))
    ref_file = rgb_files[ref_idx]
    ref_rgb = rgb_stack[ref_idx]

    print("Reference (cloudiest) image:", ref_file.name, f"({cloud_fracs[ref_idx]*100:.2f}% approx cloud)")

    draw_tile_grid(ref_rgb, args.tile_size, title=f"Tile Grid Overlay\nReference: {ref_file.name}")


if __name__ == "__main__":
    main()
