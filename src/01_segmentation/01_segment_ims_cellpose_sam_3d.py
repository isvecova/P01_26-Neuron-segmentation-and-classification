#!/usr/bin/env python3
"""
3D Cellpose-SAM segmentation for .ims files.

How to use:
1) Edit the SETTINGS section below.
2) Run this script from your IDE/editor.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import tifffile
from bioio import BioImage
from cellpose import models

import skimage


# =========================
# SETTINGS (edit these)
# =========================
# INPUT_ROOT = Path(r"L:\0_Service\Sarah\Croatia project")
INPUT_ROOT = Path(r"D:\OneDrive - IEM\000_inbox\260604_Kralova_python_troubleshooting\cellpose + far-red channel\test_mine")
OUTPUT_ROOT = INPUT_ROOT / "masks_cellpose_sam_3d_Algernon"

# Channel selection: second channel = index 1
CHANNEL_INDEX = 1

# Skip .ims filenames containing this token (case-insensitive)
SKIP_NAME_TOKEN = "Overview"

# Cellpose settings
CELLPROB_THRESHOLD = -5.0
USE_GPU = True
MODEL_TYPE = "sam"


def find_ims_files(root: Path, skip_token: str) -> List[Path]:
    # Crawl all subfolders so acquisition batches can be processed in one run.
    files: List[Path] = []
    token = skip_token.lower()
    for path in root.rglob("*.ims"):
        if token and token in path.name.lower():
            continue
        files.append(path)
    return sorted(files, reverse=True)


def extract_channel_volumes_tzyx(image: BioImage, channel_index: int) -> np.ndarray:
    """
    Extract one channel as float32 array with shape (T, Z, Y, X).
    """
    if image.scenes:
        image.set_scene(image.scenes[0])

    arr = image.get_image_data("TZYX", C=channel_index)
    arr = np.asarray(arr)

    if arr.ndim != 4:
        raise ValueError(f"Expected shape (T, Z, Y, X), got {arr.shape}")

    # Cast to native float32 early so downstream ufuncs do not receive
    # non-native-endian dtypes (e.g. ">u2") from microscopy readers.
    return arr.astype(np.float32, copy=False)


def median_filter(volume_zyx: np.ndarray) -> np.ndarray:
    """
    Preprocess a single 3D volume (Z, Y, X) for Cellpose.
    """

    footprint = skimage.morphology.disk(3)
    for z in range(volume_zyx.shape[0]):
        # rank.median expects uint8/uint16; perform explicit scaling and cast
        # to avoid implicit conversion warnings and precision surprises.
        slice_01 = skimage.exposure.rescale_intensity(volume_zyx[z], in_range="image", out_range=(0.0, 1.0))
        slice_u8 = skimage.util.img_as_ubyte(slice_01)
        filtered_u8 = skimage.filters.rank.median(slice_u8, footprint=footprint)
        volume_zyx[z] = filtered_u8.astype(np.float32, copy=False)
    return volume_zyx


def run_cellpose_3d(volumes_tzyx: np.ndarray, model: models.CellposeModel, cellprob_threshold: float) -> np.ndarray:
    """
    Run Cellpose-SAM in 3D per timepoint.
    Input:  (T, Z, Y, X)
    Output: (T, Z, Y, X) int32 masks
    """
    
    if volumes_tzyx.ndim != 4:
        raise ValueError(f"Expected (T, Z, Y, X), got {volumes_tzyx.shape}")

    # Segment each timepoint independently
    masks_per_t = []
    for time_index in range(volumes_tzyx.shape[0]):
        volume_zyx = volumes_tzyx[time_index]

        # Light denoising improves mask continuity on noisy slices.
        volume_zyx = median_filter(volume_zyx)

        # Downscale XY for faster model inference; Z is kept unchanged.
        volume_zyx_downscaled = skimage.transform.downscale_local_mean(volume_zyx, (1, 4, 4))
        masks_zyx_downscaled, _flows, _styles = model.eval(
            volume_zyx_downscaled,
            channels=[0, 0],
            do_3D=True,
            z_axis=0,
            cellprob_threshold=cellprob_threshold
        )
        masks_zyx_downscaled = np.asarray(masks_zyx_downscaled)
        if masks_zyx_downscaled.ndim != 3:
            raise ValueError(f"Expected 3D mask (Z, Y, X), got {masks_zyx_downscaled.shape}")

        # Restore masks to original resolution with nearest-neighbour interpolation
        # so label IDs stay integers and are not blended.
        masks_zyx = skimage.transform.resize(masks_zyx_downscaled, volume_zyx.shape, order=0, preserve_range=True).astype(np.int32)
        masks_per_t.append(masks_zyx.astype(np.int32, copy=False))

    return np.stack(masks_per_t, axis=0)


def save_masks(masks_tzyx: np.ndarray, src_file: Path, input_root: Path, output_root: Path) -> Path:
    # Preserve input folder structure under OUTPUT_ROOT for traceability.
    rel_parent = src_file.parent.relative_to(input_root)
    out_dir = output_root / rel_parent
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{src_file.stem}_mask_3d.tif"
    tifffile.imwrite(out_path, masks_tzyx, photometric="minisblack")
    return out_path


def main() -> None:
    if not INPUT_ROOT.exists() or not INPUT_ROOT.is_dir():
        raise FileNotFoundError(f"INPUT_ROOT does not exist or is not a directory: {INPUT_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    ims_files = find_ims_files(INPUT_ROOT, SKIP_NAME_TOKEN)
    if not ims_files:
        print("No eligible .ims files found.")
        return

    print(f"Found {len(ims_files)} eligible .ims file(s).")
    print(f"Loading Cellpose model: {MODEL_TYPE} (GPU={USE_GPU})")
    model = models.CellposeModel(gpu=USE_GPU, model_type=MODEL_TYPE)

    for i, ims_path in enumerate(ims_files, start=1):
        print(f"[{i}/{len(ims_files)}] Processing {ims_path}")

        # Skip files that already have saved masks.
        out_path = OUTPUT_ROOT / ims_path.parent.relative_to(INPUT_ROOT) / f"{ims_path.stem}_mask_3d.tif"
        if out_path.exists():
            print(f"  Skipping (output already exists)")
            continue
        try:
            # 1) Read channel data from BioImage
            bio = BioImage(str(ims_path))
            volumes_tzyx = extract_channel_volumes_tzyx(bio, CHANNEL_INDEX)

            # Save the median-filtered input volume for debugging and visualization purposes.
            debug_input_path = OUTPUT_ROOT / ims_path.parent.relative_to(INPUT_ROOT) / f"{ims_path.stem}_input_channel.tif"
            debug_input_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(debug_input_path, volumes_tzyx, photometric="minisblack")

            # 2) Run segmentation and save label volume
            masks_tzyx = run_cellpose_3d(volumes_tzyx, model, CELLPROB_THRESHOLD)
            out_path = save_masks(masks_tzyx, ims_path, INPUT_ROOT, OUTPUT_ROOT)
            print(f"  Saved: {out_path}")
        except Exception as exc:
            # Continue with the next file to avoid failing the whole batch.
            print(f"  ERROR: {exc}")

    print("Done.")


if __name__ == "__main__":
    main()
