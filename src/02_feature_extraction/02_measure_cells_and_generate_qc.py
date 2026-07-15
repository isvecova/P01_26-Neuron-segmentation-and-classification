#!/usr/bin/env python3
"""
Batch measurement script: loads masks and raw images, measures per-cell properties,
applies size-based filtering, and generates QC histograms and overview images.
"""

from bioio import BioImage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
import skimage
import skimage.transform
import skimage.segmentation

# --- Paths ---
INPUT_ROOT = Path(r"N:\Sarah\CHST11KO&P301S project\DAPI, 488 NeuN, 594 AT8, 647 WFA")   # root folder containing the original .ims images
MASK_ROOT = INPUT_ROOT / "segmentation_output_pseudo3d_260611"  # root folder containing the 3-D segmentation masks
HISTOGRAM_ROOT = INPUT_ROOT / "histograms"                 # output folder for histograms and CSVs
OVERVIEWS_ROOT = INPUT_ROOT / "overviews"                  # output folder for MIP overlay TIFFs

MASK_SUFFIX = "_mask_3d"  # suffix that distinguishes mask .tif files from other .tif files

# --- Size filter thresholds (in µm³) applied to exclude artefacts / debris ---
min_area = 300
max_area = 4000

def find_mask_files(root: Path, token: str) -> List[Path]:
    """Recursively find all .tif files under `root` whose name contains `token`."""
    # Search all subfolders to match the segmentation output structure.
    files: List[Path] = []
    for path in root.rglob("*.tif"):
        if token not in path.name.lower():
            continue
        files.append(path)
    return sorted(files)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Linearly rescale a 2-D float array to the [0, 255] range and return uint8."""
    # Fixed contrast window tuned for NeuN/AT8 channels (empirically determined).
    arr = arr.astype(np.float32)
    lo = 50
    hi = 1000
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255
    return np.clip(arr, 0, 255).astype(np.uint8)


def save_histogram(values, parameter_name, name, out_dir=HISTOGRAM_ROOT, bins=100):
    """Save a histogram PNG of `values` to `out_dir`."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]  # drop NaN / Inf before plotting

    plt.figure(figsize=(7, 4.5))
    plt.hist(values, bins=bins, edgecolor="black", alpha=0.85)
    plt.title(f"Histogram: {name}")
    plt.xlabel(parameter_name)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path / f"{name}_histogram.png", dpi=200)
    plt.close()


def main() -> None:
    # Validate that both input folders exist before starting batch processing.
    if not INPUT_ROOT.exists() or not INPUT_ROOT.is_dir():
        raise FileNotFoundError(f"INPUT_ROOT does not exist or is not a directory: {INPUT_ROOT}")
    if not MASK_ROOT.exists() or not MASK_ROOT.is_dir():
        raise FileNotFoundError(f"MASK_ROOT does not exist or is not a directory: {MASK_ROOT}")

    mask_files = find_mask_files(MASK_ROOT, MASK_SUFFIX)
    if not mask_files:
        print("No eligible .tif mask files found.")
        return

    print(f"Found {len(mask_files)} eligible mask file(s).")

    # Accumulates one summary row per image; saved to summary.csv after the loop
    records = []

    for i, mask_path in enumerate(mask_files, start=1):
        print(f"[{i}/{len(mask_files)}] Processing {mask_path}")

        # --- 1. Load segmentation mask ---
        img_mask = BioImage(mask_path)
        img_mask.set_scene(scene_id=0)
        mask_data = img_mask.get_image_data().squeeze()  # squeeze to (Z, Y, X)

        # --- 2. Derive the matching .ims image path ---
        # Strip the mask suffix and mirror the subfolder structure to locate the raw image.
        mask_name = mask_path.stem
        image_name = mask_name.replace(MASK_SUFFIX, "")
        rel = mask_path.parent.relative_to(MASK_ROOT)
        image_path = INPUT_ROOT / rel / f"{image_name}.ims"

        # --- 3. Load the corresponding raw image ---
        # Select channels 1 and 2 (NeuN=green, AT8=red) for intensity measurements.
        img = BioImage(image_path)
        img.set_scene(scene_id=0)
        image_data = img.get_image_data("CZYX", C=[1, 2])  # shape: (2, Z, Y, X)

        # Extract physical voxel size to compute calibrated measurements (e.g., area in µm³).
        xy_spacing = img.metadata.images[0].pixels.physical_size_x
        z_spacing = img.metadata.images[0].pixels.physical_size_z
        print(f"  xy_spacing={xy_spacing}, z_spacing={z_spacing}")

        green_channel = image_data[0]  # NeuN
        red_channel   = image_data[1]  # AT8

        # --- 4. Measure region properties for each segmented cell ---
        # Compute morphology + intensity metrics; spacing calibrates area to µm³.
        props_green = skimage.measure.regionprops(mask_data, green_channel, spacing=(z_spacing, xy_spacing, xy_spacing))
        props_red   = skimage.measure.regionprops(mask_data, red_channel,   spacing=(z_spacing, xy_spacing, xy_spacing))

        # Pre-compute validity flags based on size thresholds for downstream filtering.
        areas = np.array([p.area for p in props_green])
        valid = (areas >= min_area) & (areas <= max_area)

        # --- 5. Export per-region measurements to CSV (one row per cell) ---
        props_df = pd.DataFrame({
            "label":                [p.label         for p in props_green],
            "area":                 [p.area          for p in props_green],  # calibrated (µm³)
            "num_pixels":           [p.num_pixels     for p in props_green],  # raw voxel count
            "intensity_mean_green": [p.intensity_mean for p in props_green],
            "intensity_mean_red":   [p.intensity_mean for p in props_red],
            "intensity_std_red":    [p.intensity_std  for p in props_red],
            "valid":                valid,
        })
        HISTOGRAM_ROOT.mkdir(parents=True, exist_ok=True)
        props_df.to_csv(HISTOGRAM_ROOT / f"{image_name}_props.csv", index=False)

        # --- 6. Generate distribution histograms for visual QC ---
        save_histogram(props_df["area"],                 "area", f"{image_name}_area")
        save_histogram(props_df["num_pixels"],           "num_pixels", f"{image_name}_num_pixels")
        save_histogram(props_df["intensity_mean_green"], "intensity_mean_green", f"{image_name}_intensity_mean_green_channel")
        save_histogram(props_df["intensity_mean_red"],   "intensity_mean_red", f"{image_name}_intensity_mean_red_channel")
        save_histogram(props_df["intensity_std_red"],    "intensity_std_red", f"{image_name}_intensity_std_red_channel")

        # --- 7. Apply size-based filtering for visualization ---
        # Remove cells outside [min_area, max_area] from the mask for cleaner overlays.
        invalid_labels = props_df.loc[(areas < min_area) | (areas > max_area), "label"].to_numpy()
        filtered_mask_data = mask_data.copy()
        filtered_mask_data[np.isin(filtered_mask_data, invalid_labels)] = 0

        # --- 8. Generate downscaled MIP overlay for quick visual inspection ---
        # Max-project each channel and the filtered mask along Z.
        green_mip = np.asarray(green_channel).max(axis=0)  # (Y, X)
        red_mip   = np.asarray(red_channel).max(axis=0)
        mask_mip  = np.asarray(filtered_mask_data).max(axis=0).astype(np.int32)

        # Downscale by 8× to reduce file size; use nearest-neighbor for mask to keep integer labels.
        scale = 1 / 8
        green_mip_small = skimage.transform.rescale(green_mip, scale, anti_aliasing=True,  preserve_range=True)
        red_mip_small   = skimage.transform.rescale(red_mip,   scale, anti_aliasing=True,  preserve_range=True)
        mask_mip_small  = skimage.transform.rescale(mask_mip,  scale, anti_aliasing=False, order=0, preserve_range=True).astype(np.int32)

        # Compose RGB overlay: red channel → R, green channel → G, mask boundaries → white.
        boundaries = skimage.segmentation.find_boundaries(mask_mip_small, mode="outer")
        rgb = np.zeros((*green_mip_small.shape, 3), dtype=np.uint8)
        rgb[..., 0] = _to_uint8(red_mip_small)    # R = AT8
        rgb[..., 1] = _to_uint8(green_mip_small)  # G = NeuN
        rgb[boundaries] = [255, 255, 255]          # white mask outlines

        OVERVIEWS_ROOT.mkdir(parents=True, exist_ok=True)
        plt.imsave(OVERVIEWS_ROOT / f"{image_name}_mip_overlay.png", rgb)

        # --- 9. Compute summary statistics for this image ---
        # Otsu/triangle thresholds on area distribution (informational; not used for filtering).
        threshold_otsu     = skimage.filters.threshold_otsu(areas)
        threshold_triangle = skimage.filters.threshold_triangle(areas)

        n_masks       = len(props_green)
        n_valid_masks = n_masks - len(invalid_labels)

        records.append({
            "image_name":         image_name,
            "n_masks":            n_masks,
            "n_valid_masks":      n_valid_masks,
            "threshold_otsu":     threshold_otsu,
            "threshold_triangle": threshold_triangle,
            "xy_spacing":         xy_spacing,
            "z_spacing":          z_spacing,
        })

    # --- 10. Export cross-image summary CSV ---
    df = pd.DataFrame(records)
    out_csv = HISTOGRAM_ROOT / "summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSummary saved to {out_csv}")
    print("Done.")


if __name__ == "__main__":
    main()


