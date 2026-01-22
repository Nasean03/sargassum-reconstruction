"""
multi_image_reconstruct_public.py

Multi-date, tile-based cloud removal and Sargassum reconstruction
using RGB-only cloud + haze masking.

PUBLIC VERSION NOTES:
- Raw Sentinel-2 data is NOT included in this repository.
- This script accepts input/output paths via CLI arguments.
- QA60 is optional for matching (RGB+FAI is sufficient).
- GeoTIFF saving is disabled by default (enable with --save-geotiff).
"""

from __future__ import annotations

from pathlib import Path
import argparse
import re
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ========================================================================
# DEFAULT CONFIGURATION (override via CLI)
# ========================================================================

DEFAULT_TILE_SIZE = 16
DEFAULT_SARGASSUM_THRESHOLD = 0.002

# Tile-level rules (Option A – balanced)
DEFAULT_CLOUDY_TILE_THRESHOLD = 0.20
DEFAULT_CLEAR_MARGIN = 0.15
DEFAULT_MID_CLOUD_GATE = 0.60
DEFAULT_BRIGHT_TOL = 0.12
DEFAULT_COLOR_TOL = 0.20

# Sentinel-2 resolution (RGB 10m)
PIXEL_SIZE_M = 10


# ========================================================================
# CONTRAST ENHANCEMENT (DISPLAY ONLY)
# ========================================================================

def enhance_contrast(rgb: np.ndarray) -> np.ndarray:
    """Auto-stretch using 2–98% percentiles. Used for display ONLY."""
    p2 = np.percentile(rgb, 2)
    p98 = np.percentile(rgb, 98)
    stretched = (rgb - p2) / (p98 - p2 + 1e-6)
    return np.clip(stretched, 0, 1)


# ========================================================================
# FIND IMAGE SETS
# ========================================================================

def extract_date(path: Path) -> str | None:
    m = re.search(r"(\d{4}_\d{2}_\d{2})", path.name)
    return m.group(1) if m else None


def find_image_sets(raw_dir: Path, require_qa60: bool):
    """
    Match scenes by date.

    Public-safe behavior:
    - If require_qa60 is False: match RGB + FAI pairs.
    - If require_qa60 is True: match RGB + FAI + QA60 triplets.
    """
    rgb_files = sorted(list(raw_dir.glob("*_RGB.tif")) + list(raw_dir.glob("*_RGB.tiff")))
    fai_files = sorted(list(raw_dir.glob("*_FAI.tif")) + list(raw_dir.glob("*_FAItif.tif")))
    qa_files = sorted(list(raw_dir.glob("*_QA60.tif")) + list(raw_dir.glob("*_QA60.tiff")))

    def map_by_date(files):
        out = {}
        for f in files:
            d = extract_date(f)
            if d:
                out[d] = f
        return out

    rgb = map_by_date(rgb_files)
    fai = map_by_date(fai_files)
    qa = map_by_date(qa_files)

    if require_qa60:
        dates = sorted(set(rgb) & set(fai) & set(qa))
        if not dates:
            print("No complete RGB+FAI+QA60 triplets found.")
            return [], [], []
        print("\nMatched image sets (RGB <-> FAI <-> QA60):")
        R, F, Q = [], [], []
        for d in dates:
            print(f"  {rgb[d].name} <-> {fai[d].name} <-> {qa[d].name}")
            R.append(rgb[d])
            F.append(fai[d])
            Q.append(qa[d])
        return R, F, Q

    # default: RGB+FAI only
    dates = sorted(set(rgb) & set(fai))
    if not dates:
        print("No complete RGB+FAI pairs found.")
        return [], [], []
    print("\nMatched image sets (RGB <-> FAI):")
    R, F = [], []
    for d in dates:
        print(f"  {rgb[d].name} <-> {fai[d].name}")
        R.append(rgb[d])
        F.append(fai[d])
    return R, F, []


# ========================================================================
# LOADING
# ========================================================================

def is_blank_image(rgb: np.ndarray, thr: float = 0.03) -> bool:
    std = float(rgb.mean(axis=2).std())
    print(f"  Image STD = {std:.4f}")
    return std < thr


def load_rgb_as_float(path: Path):
    with rasterio.open(path) as src:
        rgb = src.read([1, 2, 3]).astype(np.float32)
        profile = src.profile
    rgb = np.transpose(rgb, (1, 2, 0))
    return np.clip(rgb / 10000.0, 0, 1), profile


def load_fai(path: Path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


# ========================================================================
# CLOUD MASK OPERATIONS
# ========================================================================

def compute_cloud_mask_from_rgb(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    brightness = (r + g + b) / 3

    maxc = np.max(rgb, axis=2)
    minc = np.min(rgb, axis=2)
    sat = (maxc - minc) / (brightness + 1e-6)

    bright_cloud = (brightness > 0.40) & (sat < 0.35)

    haze_cloud = (
        (0.15 < brightness) & (brightness < 0.45) &
        (sat < 0.22) &
        (b > r) & (b > g)
    )

    return (bright_cloud | haze_cloud).astype(np.uint8)


def compute_tile_cloud_fraction(mask: np.ndarray, tile_size: int) -> np.ndarray:
    """
    Compute cloud fraction per tile. Supports partial edge tiles by using
    ceil division and safe slicing.
    """
    H, W = mask.shape
    rows = int(np.ceil(H / tile_size))
    cols = int(np.ceil(W / tile_size))

    out = np.zeros((rows, cols), dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            y0 = i * tile_size
            x0 = j * tile_size
            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)
            t = mask[y0:y1, x0:x1]
            out[i, j] = t.mean() if t.size > 0 else 0.0

    return out


# ========================================================================
# RECONSTRUCTION
# ========================================================================

def reconstruct(
    rgb_stack: np.ndarray,
    fai_stack: np.ndarray,
    tile_clouds: list[np.ndarray],
    ref_index: int,
    tile_size: int,
    sargassum_threshold: float,
    cloudy_tile_threshold: float,
    clear_margin: float,
    mid_cloud_gate: float,
    bright_tol: float,
    color_tol: float,
):
    """
    Reconstruct reference image using donor tiles from other images.
    Works with partial tiles at edges.
    """
    H, W, _ = rgb_stack[ref_index].shape
    n_img = len(rgb_stack)
    rows = int(np.ceil(H / tile_size))
    cols = int(np.ceil(W / tile_size))

    reconstructed = rgb_stack[ref_index].copy()
    source_map = -1 * np.ones((rows, cols), dtype=np.int32)

    for i in range(rows):
        for j in range(cols):
            ref_cloud = tile_clouds[ref_index][i, j]
            if ref_cloud < cloudy_tile_threshold:
                continue

            y0 = i * tile_size
            x0 = j * tile_size
            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)

            ref_tile = rgb_stack[ref_index][y0:y1, x0:x1]
            ref_bright = float(ref_tile.mean())

            best = None
            best_cloud = None
            best_sarg = False

            for k in range(n_img):
                if k == ref_index:
                    continue

                donor_cloud = tile_clouds[k][i, j]

                # require donor to be noticeably clearer than reference
                if donor_cloud >= ref_cloud - clear_margin:
                    continue

                donor_tile = rgb_stack[k][y0:y1, x0:x1]
                donor_bright = float(donor_tile.mean())

                if donor_tile.shape != ref_tile.shape:
                    continue

                if ref_cloud < mid_cloud_gate:
                    if abs(donor_bright - ref_bright) > bright_tol:
                        continue
                    if np.mean(np.abs(donor_tile - ref_tile)) > color_tol:
                        continue

                fai_tile = fai_stack[k][y0:y1, x0:x1]
                has_sarg = (fai_tile > sargassum_threshold).mean() > 0.05 if fai_tile.size > 0 else False

                if best is None:
                    best = k
                    best_cloud = donor_cloud
                    best_sarg = has_sarg
                else:
                    if has_sarg and not best_sarg:
                        best = k
                        best_cloud = donor_cloud
                        best_sarg = True
                    elif has_sarg == best_sarg and donor_cloud < best_cloud:
                        best = k
                        best_cloud = donor_cloud
                        best_sarg = has_sarg

            if best is not None:
                reconstructed[y0:y1, x0:x1] = rgb_stack[best][y0:y1, x0:x1]
                source_map[i, j] = best

    return reconstructed, source_map


# ========================================================================
# SAVE + DISPLAY
# ========================================================================

def save_rgb_geotiff(path: Path, rgb: np.ndarray, profile: dict):
    H, W, _ = rgb.shape
    prof = profile.copy()
    prof.update(count=3, dtype=rasterio.uint16)

    data = np.clip(rgb * 10000.0, 0, 10000).astype(np.uint16)
    data = np.transpose(data, (2, 0, 1))

    with rasterio.open(path, "w", **prof) as dst:
        dst.write(data)


def show_full_comparison(
    ref_rgb: np.ndarray,
    rec_rgb: np.ndarray,
    other_images: list[np.ndarray],
    other_names: list[str],
    title: str,
    total_tiles: int,
    replaced_tiles: int,
    contributions: dict[str, int],
    tile_size: int,
):
    tile_m = tile_size * PIXEL_SIZE_M
    tile_label = f"{tile_size} px ({tile_m} m × {tile_m} m)"

    ref_rgb = enhance_contrast(ref_rgb)
    rec_rgb = enhance_contrast(rec_rgb)
    other_images = [enhance_contrast(img) for img in other_images]

    num_others = len(other_images)

    fig = plt.figure(figsize=(18, 12 + 2 * (num_others > 0)))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.3, 0.8, 0.7], width_ratios=[1, 1], hspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(ref_rgb)
    ax1.set_title("REFERENCE IMAGE\n" + title, fontsize=17, weight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(rec_rgb)
    ax2.set_title(f"RECONSTRUCTED IMAGE\nTile size: {tile_label}", fontsize=17, weight="bold")
    ax2.axis("off")

    if num_others > 0:
        cols = num_others
        gs2 = gridspec.GridSpecFromSubplotSpec(1, cols, subplot_spec=gs[1, :], wspace=0.35, hspace=0.25)

        ax_heading = fig.add_subplot(gs[1, :])
        ax_heading.axis("off")
        ax_heading.set_title("\n\nDONOR IMAGES", fontsize=14, weight="bold")

        for i in range(num_others):
            ax = fig.add_subplot(gs2[0, i])
            ax.imshow(other_images[i])
            name_short = other_names[i].replace("Barbados_", "").replace("_RGB.tif", "").replace("_RGB.tiff", "")
            ax.set_title(name_short, fontsize=8, pad=6)
            ax.axis("off")

    ax_text = fig.add_subplot(gs[2, :])
    ax_text.axis("off")

    pct = (replaced_tiles / total_tiles) * 100 if total_tiles > 0 else 0.0
    text_lines = [
        f"Total tiles: {total_tiles}",
        f"Replaced tiles: {replaced_tiles} ({pct:.2f}%)",
        "",
        "Tile contributions:",
    ]
    for name, count in contributions.items():
        text_lines.append(f"  {name} -> {count} tiles")

    ax_text.text(0.02, 0.98, "\n".join(text_lines), fontsize=13, va="top", family="monospace")

    plt.tight_layout()
    plt.show()


# ========================================================================
# MAIN
# ========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, required=True, help="Folder with *_RGB.tif(f) and *_FAI.tif (not included).")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"), help="Where to save previews/maps.")
    ap.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    ap.add_argument("--require-qa60", action="store_true", help="Require QA60 files for matching (optional).")

    ap.add_argument("--sargassum-threshold", type=float, default=DEFAULT_SARGASSUM_THRESHOLD)
    ap.add_argument("--cloudy-tile-threshold", type=float, default=DEFAULT_CLOUDY_TILE_THRESHOLD)
    ap.add_argument("--clear-margin", type=float, default=DEFAULT_CLEAR_MARGIN)
    ap.add_argument("--mid-cloud-gate", type=float, default=DEFAULT_MID_CLOUD_GATE)
    ap.add_argument("--bright-tol", type=float, default=DEFAULT_BRIGHT_TOL)
    ap.add_argument("--color-tol", type=float, default=DEFAULT_COLOR_TOL)

    ap.add_argument("--save-geotiff", action="store_true", help="Enable GeoTIFF saving (off by default).")
    ap.add_argument("--save-source-map", action="store_true", help="Enable saving source index map .npy (off by default).")
    args = ap.parse_args()

    if not args.raw_dir.exists():
        raise RuntimeError("Input directory not found. Raw data is not included in this repository.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    RGB, FAI, _QA = find_image_sets(args.raw_dir, require_qa60=args.require_qa60)
    if not RGB:
        return

    rgb_stack_list, fai_stack_list = [], []
    profile = None

    for r, f in zip(RGB, FAI):
        print("\nLoading", r.name)
        rgb, prof = load_rgb_as_float(r)
        rgb_stack_list.append(rgb)
        if profile is None:
            profile = prof

        print("Loading", f.name)
        fai_stack_list.append(load_fai(f))

    rgb_stack = np.array(rgb_stack_list)
    fai_stack = np.array(fai_stack_list)

    # Remove blank scenes
    keep = []
    for i, rgb in enumerate(rgb_stack):
        print("\nChecking", RGB[i].name)
        if not is_blank_image(rgb):
            keep.append(i)
        else:
            print("  -> discarded")

    if not keep:
        print("All images discarded.")
        return

    rgb_stack = rgb_stack[keep]
    fai_stack = fai_stack[keep]
    RGB = [RGB[i] for i in keep]

    n = len(rgb_stack)
    print("\nRemaining usable images:", n)

    # Cloud maps
    tile_clouds = []
    frac = []
    for i in range(n):
        print("Cloud mask for", RGB[i].name)
        m = compute_cloud_mask_from_rgb(rgb_stack[i])
        tc = compute_tile_cloud_fraction(m, args.tile_size)
        tile_clouds.append(tc)
        frac.append(tc.mean())

    frac = np.array(frac)

    # Cloudiest image = reference
    REF = int(np.argmax(frac))

    print("\nReference selection:")
    for i, v in enumerate(frac):
        print(f"  {RGB[i].name}: {v*100:.2f}% cloudy")
    print("\nUsing reference:", RGB[REF].name)

    ref = rgb_stack[REF]
    rec, source_map = reconstruct(
        rgb_stack, fai_stack, tile_clouds, REF,
        tile_size=args.tile_size,
        sargassum_threshold=args.sargassum_threshold,
        cloudy_tile_threshold=args.cloudy_tile_threshold,
        clear_margin=args.clear_margin,
        mid_cloud_gate=args.mid_cloud_gate,
        bright_tol=args.bright_tol,
        color_tol=args.color_tol,
    )

    total = source_map.size
    replaced = int(np.sum(source_map != -1))

    print(f"\nTotal tiles: {total}")
    print(f"Replaced: {replaced} ({(replaced/total)*100:.2f}%)" if total > 0 else f"Replaced: {replaced}")

    # Residual Cloud Coverage (RCC)
    rcc = compute_cloud_mask_from_rgb(rec).mean() * 100
    print(f"Residual Cloud Coverage (RCC): {rcc:.2f}%")

    contrib = {}
    print("\nTile contributions:")
    for k in range(n):
        used = int(np.sum(source_map == k))
        if used > 0:
            contrib[RGB[k].name] = used
            print(f"  {RGB[k].name}: {used}")

    # Outputs (public-safe defaults)
    ref_date = RGB[REF].stem.replace("_RGB", "")
    out_png = args.out_dir / f"{ref_date}_reconstructed.png"

    print("Saving bright PNG preview ->", out_png)
    plt.imsave(out_png, enhance_contrast(rec))

    if args.save_geotiff:
        out_tif = args.out_dir / f"{ref_date}_reconstructed.tif"
        print("Saving GeoTIFF ->", out_tif)
        save_rgb_geotiff(out_tif, rec, profile)
    else:
        print("GeoTIFF saving disabled (use --save-geotiff to enable).")

    if args.save_source_map:
        out_map = args.out_dir / f"{ref_date}_source_index_map.npy"
        print("Saving source map ->", out_map)
        np.save(out_map, source_map)
    else:
        print("Source map saving disabled (use --save-source-map to enable).")

    # Prepare other images for display
    other_images = []
    other_names = []
    for i in range(n):
        if i == REF:
            continue
        other_images.append(rgb_stack[i])
        other_names.append(RGB[i].name)

    show_full_comparison(
        ref, rec,
        other_images, other_names,
        f"({ref_date})",
        total, replaced, contrib,
        tile_size=args.tile_size,
    )

    print("\nReconstruction complete.\n")


if __name__ == "__main__":
    main()
