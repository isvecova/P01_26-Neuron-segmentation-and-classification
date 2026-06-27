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

import pandas as pd
from datetime import datetime


# =========================
# SETTINGS (edit these)
# =========================
# INPUT_ROOT = Path(r"L:\0_Service\Sarah\Croatia project")
INPUT_ROOT = Path(r"D:\OneDrive - IEM\000_inbox\260604_Kralova_python_troubleshooting\cellpose + far-red channel\data")
OUTPUT_ROOT = Path(r"L:\Algernon\530\Sarah\segmentation_output_pseudo3d_260611")

# Channel selection: second channel = index 1
CHANNEL_INDEX = 1

# Skip .ims filenames containing this token (case-insensitive)
SKIP_NAME_TOKEN = "Overview"

# Cellpose settings
CELLPROB_THRESHOLD = -2
ANISOTROPY = 1.5
STITCH_THRESOLD = 0.1
USE_GPU = True
MODEL_PATH = r"N:\03_analysis_tools\Sarah_cellpose_training_script\models\260610_cellpose_Kralova_retrained_1"

NON_BINNED_STEP = 0.152
# Comment: The dataset contains both binned and non-binned images. The non-binned images have a pixel size of 0.152 µm, while the binned images have a pixel size of 0.304 µm. 
# The script will automatically downscale the non-binned images to match the binned resolution for consistent processing.


def inspect_existing_mask(mask_path: Path) -> tuple[bool, str, tuple | None]:
    # The previous version of the script has an issue with segmenting non-binned images due to mistake in downscaling. 
    # This script will check the existing mask files and determine if they are faulty (e.g., missing, empty, or have unexpected dimensions) before deciding to rerun the segmentation.
    
    """
    Returns:
        is_faulty, reason, shape
    """
    if not mask_path.exists():
        return True, "missing", None

    try:
        arr = tifffile.imread(mask_path)
        shape = arr.shape

        if arr.ndim == 4:
            return True, "faulty_4d_mask", shape

        if arr.ndim != 3:
            return True, f"faulty_unexpected_ndim_{arr.ndim}", shape

        if arr.max() == 0:
            return True, "faulty_empty_mask", shape

        return False, "ok", shape

    except Exception as exc:
        return True, f"faulty_unreadable_mask: {exc}", None

def find_ims_files(root: Path, skip_token: str) -> List[Path]:
    # Crawl all subfolders so acquisition batches can be processed in one run.
    files: List[Path] = []
    token = skip_token.lower()
    for path in root.rglob("*.ims"):
        if token and token in path.name.lower():
            continue
        files.append(path)
    return sorted(files, reverse=True)


def extract_channel_volumes_zyx(image: BioImage, channel_index: int) -> np.ndarray:
    """
    Extract one channel as float32 array with shape (T, Z, Y, X).
    """
    if image.scenes:
        image.set_scene(image.scenes[0])

    arr = image.get_image_data("ZYX", C=channel_index, T=0)
    arr = np.asarray(arr)

    if arr.ndim != 3:
        raise ValueError(f"Expected shape (Z, Y, X), got {arr.shape}")

    x_step = image.metadata.images[0].pixels.physical_size_x
    downscaled = False
    if x_step is not None and np.abs(x_step - NON_BINNED_STEP) < 0.1:
        arr = skimage.transform.downscale_local_mean(arr, (1, 2, 2))
        downscaled = True
        print(f"Rescaled from x_step={x_step:.3f} to 2x2 binning.")

    # Cast to native float32 early so downstream ufuncs do not receive
    # non-native-endian dtypes (e.g. ">u2") from microscopy readers.
    return arr.astype(np.float32, copy=False), downscaled


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


def run_cellpose_3d(volume_zyx: np.ndarray, model: models.CellposeModel, cellprob_threshold: float) -> np.ndarray:
    """
    Run Cellpose-SAM in 3D.
    Input:  (Z, Y, X)
    Output: (Z, Y, X) int32 masks
    """
    
    if volume_zyx.ndim != 3:
        raise ValueError(f"Expected (Z, Y, X), got {volume_zyx.shape}")

    # # Light denoising improves mask continuity on noisy slices.
    # volume_zyx = median_filter(volume_zyx)

    # Downscale XY for faster model inference; Z is kept unchanged.
    masks_zyx_downscaled, _flows, _styles = model.eval(
        volume_zyx,
        channels=[0, 0],
        stitch_threshold=STITCH_THRESOLD,
        anisotropy=ANISOTROPY,
        cellprob_threshold=cellprob_threshold,
        z_axis=0
    )
    masks_zyx_downscaled = np.asarray(masks_zyx_downscaled)
    if masks_zyx_downscaled.ndim != 3:
        raise ValueError(f"Expected 3D mask (Z, Y, X), got {masks_zyx_downscaled.shape}")

    # Restore masks to original resolution with nearest-neighbour interpolation
    # so label IDs stay integers and are not blended.
    if masks_zyx_downscaled.shape != volume_zyx.shape:
        masks_zyx = skimage.transform.resize(
            masks_zyx_downscaled,
            volume_zyx.shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.int32)
    else:
        masks_zyx = masks_zyx_downscaled.astype(np.int32)
    return masks_zyx.astype(np.int32, copy=False)


def save_masks(masks_zyx: np.ndarray, src_file: Path, input_root: Path, output_root: Path) -> Path:
    # Preserve input folder structure under OUTPUT_ROOT for traceability.
    rel_parent = src_file.parent.relative_to(input_root)
    out_dir = output_root / rel_parent
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{src_file.stem}_mask_3d.tif"
    tifffile.imwrite(out_path, masks_zyx, photometric="minisblack")
    return out_path


def main() -> None:
    if not INPUT_ROOT.exists() or not INPUT_ROOT.is_dir():
        raise FileNotFoundError(f"INPUT_ROOT does not exist or is not a directory: {INPUT_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    log_rows = []
    log_path = OUTPUT_ROOT / "segmentation_rerun_log.csv"

    ims_files = find_ims_files(INPUT_ROOT, SKIP_NAME_TOKEN)
    if not ims_files:
        print("No eligible .ims files found.")
        return

    print(f"Found {len(ims_files)} eligible .ims file(s).")
    print(f"Loading Cellpose model: {MODEL_PATH} (GPU={USE_GPU})")
    model = models.CellposeModel(gpu=USE_GPU, pretrained_model=MODEL_PATH)

    for i, ims_path in enumerate(ims_files, start=1):
        print(f"[{i}/{len(ims_files)}] Checking {ims_path}")

        rel_parent = ims_path.parent.relative_to(INPUT_ROOT)
        out_dir = OUTPUT_ROOT / rel_parent
        out_path = out_dir / f"{ims_path.stem}_mask_3d.tif"

        is_faulty, reason, old_shape = inspect_existing_mask(out_path)

        if not is_faulty:
            print(f"  Skipping: existing mask OK, shape={old_shape}")

            log_rows.append({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "input_file": str(ims_path),
                "mask_file": str(out_path),
                "action": "skipped",
                "reason": reason,
                "old_mask_shape": str(old_shape),
                "new_mask_shape": None,
                "status": "ok",
                "error": None,
            })
            continue

        print(f"  Rerunning segmentation: {reason}, old_shape={old_shape}")

        try:
            # 1) Read channel data from BioImage
            bio = BioImage(str(ims_path))
            bio.set_scene(bio.scenes[2])  # Use the 4x binned scene for faster processing
            volume_zyx, downscaled = extract_channel_volumes_zyx(bio, CHANNEL_INDEX)

            # Save the median-filtered input volume for debugging and visualization purposes.
            debug_input_path = OUTPUT_ROOT / ims_path.parent.relative_to(INPUT_ROOT) / f"{ims_path.stem}_input_channel.tif"
            debug_input_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(debug_input_path, volume_zyx, photometric="minisblack")

            # 2) Run segmentation and save label volume
            masks_zyx = run_cellpose_3d(volume_zyx, model, CELLPROB_THRESHOLD)

            # Upscale again if the input was downscaled - combined together in the run_cellpose_3d function
            if masks_zyx.ndim != 3:
                raise ValueError(f"Refusing to save non-3D mask with shape {masks_zyx.shape}")

            print(f"Final mask shape: {masks_zyx.shape}")
            print(f"Final mask dtype: {masks_zyx.dtype}")
            print(f"Number of labels: {masks_zyx.max()}")

            out_path = save_masks(masks_zyx, ims_path, INPUT_ROOT, OUTPUT_ROOT)

            print(f"  Saved corrected mask: {out_path}")

            log_rows.append({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "input_file": str(ims_path),
                "mask_file": str(out_path),
                "action": "redone",
                "reason": reason,
                "old_mask_shape": str(old_shape),
                "new_mask_shape": str(masks_zyx.shape),
                "status": "ok",
                "error": None,
            })

        except Exception as exc:
            # Continue with the next file to avoid failing the whole batch.
            print(f"  ERROR: {exc}")
            log_rows.append({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "input_file": str(ims_path),
                "mask_file": str(out_path),
                "action": "failed",
                "reason": reason,
                "old_mask_shape": str(old_shape),
                "new_mask_shape": None,
                "status": "error",
                "error": str(exc),
            })
            pd.DataFrame(log_rows).to_csv(log_path, index=False)

    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print(f"Log saved to: {log_path}")
    print("Done.")


if __name__ == "__main__":
    main()
