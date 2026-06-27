#!/usr/bin/env python3

# The script extract cell features from 3D cellpose segmentation masks and corresponding multi-channel images.
# It saves the extracted features to a CSV file and generates overview images and histograms for quality control.

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

from bioio import BioImage

import skimage
import skimage.transform
import skimage.segmentation
import skimage.filters
import skimage.measure
import skimage.exposure
import skimage.morphology

from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter
from skimage.segmentation import expand_labels
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label as cc_label, regionprops


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

INPUT_ROOT = Path(r"N:\01_scientific_data\Sarah_Kralova_brain_slice_data\cellpose + far-red channel")
MASK_ROOT = Path(r"L:\Algernon\530\Sarah\segmentation_output_pseudo3d_260611")

HISTOGRAM_ROOT = INPUT_ROOT / "histograms"
OVERVIEWS_ROOT = INPUT_ROOT / "overviews"
QC_CROPS_ROOT = INPUT_ROOT / "qc_crops"

for p in (HISTOGRAM_ROOT, OVERVIEWS_ROOT, QC_CROPS_ROOT):
    p.mkdir(parents=True, exist_ok=True)

MASK_SUFFIX = "_mask_3d"


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------

CH_DAPI = 0
CH_GREEN = 1
CH_RED = 2
CH_FARRED = 3

BIOIO_SCENE_INDEX = 0
NON_BINNED_STEP = 0.152


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

MIN_AREA = 300
MAX_AREA = 4000

INTENSITY_PERCENTILES = tuple(range(10, 100, 10)) + (92, 95, 98)

DAPI_OPENING_RADIUS = 2

# Tissue detection from green MIP
# Comment: Red channel might be more suitable, in some images, there are some green dots even in the background.
GREEN_TISSUE_BLUR_SIGMA = 10
GREEN_TISSUE_THRESHOLD = 115
GREEN_TISSUE_OPENING_RADIUS = 5
GREEN_TISSUE_CLOSING_RADIUS = 20
GREEN_TISSUE_MIN_SIZE = 50_000
GREEN_TISSUE_FILL_HOLES = True

RED_BLUR_SIGMAS_UM = [0.5, 1.0, 2.0]
MASK_DILATION_RADIUS_VOXELS = 2

PUNCTA_PERCENTILE = 99
PUNCTA_MIN_VOXELS = 3

GLCM_LEVELS = 32
GLCM_MIN_PIXELS = 50

OVERVIEW_DPI = 500
OVERLAY_SCALE = 1 / 8


# ---------------------------------------------------------------------------
# BioIO reading
# ---------------------------------------------------------------------------

def read_ims_channels_bioio(
    path: Path,
    channel_indices: List[int],
    scene_index: int | None = 0,
):
    """
    Read selected channels using BioIO.

    Returns
    -------
    data : np.ndarray
        Shape (C, Z, Y, X)
    xy_spacing : float
        XY pixel size in µm
    z_spacing : float
        Z spacing in µm
    """
    bio = BioImage(str(path))

    x_step = bio.metadata.images[0].pixels.physical_size_x
    if x_step is not None and np.abs(x_step - NON_BINNED_STEP) < 0.1:
        scene_index = 1
        print(f"  Detected non-binned image, using scene_index={scene_index}")
    else: 
        scene_index = 0
        print(f"  Detected binned image, using scene_index={scene_index}")

    if scene_index is not None:
        if scene_index >= len(bio.scenes):
            raise ValueError(
                f"Requested scene_index={scene_index}, but file has only "
                f"{len(bio.scenes)} scenes: {bio.scenes}"
            )
        bio.set_scene(bio.scenes[scene_index])

    data = bio.get_image_data("CZYX", T=0)

    pixels = bio.metadata.images[scene_index].pixels

    xy_spacing = pixels.physical_size_x
    z_spacing = pixels.physical_size_z

    xy_spacing = float(xy_spacing) if xy_spacing is not None else 1.0
    z_spacing = float(z_spacing) if z_spacing is not None else 1.0

    print(f"  Image spacing: xy={xy_spacing:.4f} µm, z={z_spacing:.4f} µm")

    # channel_indices retained for compatibility with previous function signature
    return data, xy_spacing, z_spacing


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def find_mask_files(root: Path, token: str) -> List[Path]:
    token = token.lower()
    return sorted(
        path for path in root.rglob("*.tif")
        if token in path.name.lower()
    )


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo, hi = 50, 1000
    arr = (arr - lo) / (hi - lo) * 255
    return np.clip(arr, 0, 255).astype(np.uint8)


def save_histogram(values, parameter_name, name, out_dir, bins=100):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return

    plt.figure(figsize=(7, 4.5))
    plt.hist(values, bins=bins, edgecolor="black", alpha=0.85)
    plt.title(f"Histogram: {name}")
    plt.xlabel(parameter_name)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / f"{name}_histogram.png", dpi=200)
    plt.close()


def add_feature_dict_to_props_df(props_df: pd.DataFrame, feature_dict: dict) -> pd.DataFrame:
    if not feature_dict:
        return props_df

    feature_df = (
        pd.DataFrame.from_dict(feature_dict, orient="index")
        .reset_index()
        .rename(columns={"index": "label"})
    )

    feature_df["label"] = feature_df["label"].astype(props_df["label"].dtype)

    return props_df.merge(feature_df, on="label", how="left")


def sigma_um_to_voxels(sigma_um: float, z_spacing: float, xy_spacing: float):
    return (
        sigma_um / z_spacing,
        sigma_um / xy_spacing,
        sigma_um / xy_spacing,
    )


def labels_from_mask(mask_data):
    labels = np.unique(mask_data)
    return labels[labels > 0].astype(np.uint32)


def safe_divide(a, b):
    return a / np.maximum(b, 1)


# ---------------------------------------------------------------------------
# Alignment: downscale image to mask, not mask to image
# ---------------------------------------------------------------------------

def downscale_image_to_mask_shape(image_data, mask_shape):
    """
    Downscale image_data from CZYX to match mask ZYX shape.

    Returns
    -------
    image_rescaled : np.ndarray
        Shape (C,) + mask_shape
    scale_z, scale_y, scale_x : float
        Scaling factors by which physical spacing must be multiplied.
    """
    target_shape = (image_data.shape[0],) + tuple(mask_shape)

    print(f"  Downscaling image from {image_data.shape} to {target_shape}")

    image_rescaled = skimage.transform.resize(
        image_data,
        target_shape,
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(image_data.dtype, copy=False)

    scale_z = image_data.shape[1] / mask_shape[0]
    scale_y = image_data.shape[2] / mask_shape[1]
    scale_x = image_data.shape[3] / mask_shape[2]

    if not np.isclose(scale_y, scale_x):
        print(
            f"  WARNING: Y and X scale factors differ: "
            f"scale_y={scale_y:.4f}, scale_x={scale_x:.4f}. "
            f"Using scale_y for xy_spacing."
        )

    return image_rescaled, scale_z, scale_y, scale_x


# ---------------------------------------------------------------------------
# Fast statistics
# ---------------------------------------------------------------------------

def compute_channel_statistics_from_regionprops(props_list, prefix):
    feature_dict = {}

    for prop in props_list:
        values = prop.intensity_image[prop.image]
        values = values[np.isfinite(values)]

        if values.size == 0:
            continue

        mean = float(np.mean(values))
        std = float(np.std(values))

        row = {
            f"{prefix}_mean": mean,
            f"{prefix}_median": float(np.median(values)),
            f"{prefix}_std": std,
            f"{prefix}_min": float(np.min(values)),
            f"{prefix}_max": float(np.max(values)),
            f"{prefix}_sum": float(np.sum(values)),
            f"{prefix}_cv": float(std / (mean + 1e-8)),
        }

        for percentile in INTENSITY_PERCENTILES:
            row[f"{prefix}_p{percentile}"] = float(np.percentile(values, percentile))

        feature_dict[int(prop.label)] = row

    return feature_dict


def compute_channel_statistics_from_labels_bbox(mask_data, intensity_image, prefix):
    """
    Label loop is retained only over local bounding boxes.
    This avoids full-volume mask_data == label_id scans.
    """
    feature_dict = {}
    objects = ndi.find_objects(mask_data)

    for label_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue

        local_mask = mask_data[slc] == label_id
        values = intensity_image[slc][local_mask]
        values = values[np.isfinite(values)]

        if values.size == 0:
            continue

        mean = float(np.mean(values))
        std = float(np.std(values))

        row = {
            f"{prefix}_mean": mean,
            f"{prefix}_median": float(np.median(values)),
            f"{prefix}_std": std,
            f"{prefix}_min": float(np.min(values)),
            f"{prefix}_max": float(np.max(values)),
            f"{prefix}_sum": float(np.sum(values)),
            f"{prefix}_cv": float(std / (mean + 1e-8)),
        }

        for percentile in INTENSITY_PERCENTILES:
            row[f"{prefix}_p{percentile}"] = float(np.percentile(values, percentile))

        feature_dict[int(label_id)] = row

    return feature_dict


# ---------------------------------------------------------------------------
# Fast dilated mask features
# ---------------------------------------------------------------------------

def compute_dilated_mask_features_fast(
    mask_data,
    intensity_image,
    radius=2,
    prefix="red_dilated",
):
    """
    Fast replacement for per-label binary dilation.

    Uses expand_labels(), so neighbouring expanded objects do not overlap.
    This differs from independent binary dilation, where neighbouring shells
    can overlap and the same background voxel can contribute to multiple labels.
    """
    labels = labels_from_mask(mask_data)

    if labels.size == 0:
        return {}

    expanded = expand_labels(mask_data, distance=radius).astype(mask_data.dtype, copy=False)

    shell_labels = expanded.copy()
    shell_labels[mask_data > 0] = 0

    ones = np.ones_like(intensity_image, dtype=np.uint8)

    inside_count = ndi.sum(ones, labels=mask_data, index=labels)
    inside_sum = ndi.sum(intensity_image, labels=mask_data, index=labels)
    inside_mean = safe_divide(inside_sum, inside_count)

    dilated_count = ndi.sum(ones, labels=expanded, index=labels)
    dilated_sum = ndi.sum(intensity_image, labels=expanded, index=labels)
    dilated_mean = safe_divide(dilated_sum, dilated_count)

    shell_count = ndi.sum(ones, labels=shell_labels, index=labels)
    shell_sum = ndi.sum(intensity_image, labels=shell_labels, index=labels)
    shell_mean = safe_divide(shell_sum, shell_count)

    inside_max = ndi.maximum(intensity_image, labels=mask_data, index=labels)
    dilated_max = ndi.maximum(intensity_image, labels=expanded, index=labels)
    shell_max = ndi.maximum(intensity_image, labels=shell_labels, index=labels)

    mask_objects = ndi.find_objects(mask_data)
    expanded_objects = ndi.find_objects(expanded)
    shell_objects = ndi.find_objects(shell_labels)

    feature_dict = {}

    for i, label_id in enumerate(labels):
        label_int = int(label_id)

        inside_p95 = np.nan
        dilated_p95 = np.nan
        shell_p95 = np.nan

        slc_inside = mask_objects[label_int - 1] if label_int - 1 < len(mask_objects) else None
        if slc_inside is not None:
            local = mask_data[slc_inside] == label_id
            values = intensity_image[slc_inside][local]
            if values.size:
                inside_p95 = float(np.percentile(values, 95))

        slc_dilated = expanded_objects[label_int - 1] if label_int - 1 < len(expanded_objects) else None
        if slc_dilated is not None:
            local = expanded[slc_dilated] == label_id
            values = intensity_image[slc_dilated][local]
            if values.size:
                dilated_p95 = float(np.percentile(values, 95))

        slc_shell = shell_objects[label_int - 1] if label_int - 1 < len(shell_objects) else None
        if slc_shell is not None:
            local = shell_labels[slc_shell] == label_id
            values = intensity_image[slc_shell][local]
            if values.size:
                shell_p95 = float(np.percentile(values, 95))

        row = {
            f"{prefix}_mean": float(dilated_mean[i]) if dilated_count[i] else np.nan,
            f"{prefix}_p95": dilated_p95,
            f"{prefix}_max": float(dilated_max[i]) if dilated_count[i] else np.nan,
            f"{prefix}_sum": float(dilated_sum[i]) if dilated_count[i] else np.nan,

            f"{prefix}_shell_mean": float(shell_mean[i]) if shell_count[i] else np.nan,
            f"{prefix}_shell_p95": shell_p95,
            f"{prefix}_shell_max": float(shell_max[i]) if shell_count[i] else np.nan,
            f"{prefix}_shell_sum": float(shell_sum[i]) if shell_count[i] else np.nan,

            f"{prefix}_inside_mean": float(inside_mean[i]) if inside_count[i] else np.nan,
            f"{prefix}_inside_p95": inside_p95,

            f"{prefix}_shell_to_inside_mean_ratio": (
                float(shell_mean[i] / (inside_mean[i] + 1e-8))
                if shell_count[i] and inside_count[i] else np.nan
            ),
            f"{prefix}_dilated_to_inside_mean_ratio": (
                float(dilated_mean[i] / (inside_mean[i] + 1e-8))
                if dilated_count[i] and inside_count[i] else np.nan
            ),
        }

        feature_dict[label_int] = row

    return feature_dict


# ---------------------------------------------------------------------------
# Other red-channel features
# ---------------------------------------------------------------------------

def compute_puncta_features_fast(mask_data, red_channel, percentile=99, prefix="red_puncta"):
    feature_dict = {}

    labels = labels_from_mask(mask_data)

    if labels.size == 0:
        return feature_dict

    red_positive_values = red_channel[red_channel > 0]

    if red_positive_values.size == 0:
        return feature_dict

    threshold = float(np.percentile(red_positive_values, percentile))
    high_red = red_channel > threshold

    ones = np.ones_like(red_channel, dtype=np.uint8)

    cell_voxels = ndi.sum(ones, labels=mask_data, index=labels)
    puncta_voxels = ndi.sum(high_red.astype(np.uint8), labels=mask_data, index=labels)
    puncta_sum = ndi.sum(red_channel * high_red, labels=mask_data, index=labels)
    puncta_mean = safe_divide(puncta_sum, puncta_voxels)

    for i, label_id in enumerate(labels):
        n_cell = int(cell_voxels[i])
        n_puncta = int(puncta_voxels[i])

        if n_cell == 0:
            continue

        feature_dict[int(label_id)] = {
            f"{prefix}_threshold": threshold,
            f"{prefix}_pixels": n_puncta,
            f"{prefix}_fraction": float(n_puncta / n_cell),
            f"{prefix}_mean_high_pixels": float(puncta_mean[i]) if n_puncta else 0.0,
            f"{prefix}_sum_high_pixels": float(puncta_sum[i]) if n_puncta else 0.0,
        }

    return feature_dict


def compute_puncta_structure_features_bbox(
    mask_data,
    red_channel,
    percentile=99,
    min_puncta_voxels=3,
    prefix="red_puncta_structure",
):
    feature_dict = {}

    red_positive_values = red_channel[red_channel > 0]

    if red_positive_values.size == 0:
        return feature_dict

    threshold = float(np.percentile(red_positive_values, percentile))
    objects = ndi.find_objects(mask_data)

    for label_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue

        local_mask = mask_data[slc] == label_id
        cell_voxels = int(local_mask.sum())

        if cell_voxels == 0:
            continue

        high_red = (red_channel[slc] > threshold) & local_mask

        cc = cc_label(high_red)
        props = regionprops(cc)

        puncta_sizes = np.array(
            [p.area for p in props if p.area >= min_puncta_voxels],
            dtype=float,
        )

        if puncta_sizes.size == 0:
            row = {
                f"{prefix}_n": 0,
                f"{prefix}_total_voxels": 0,
                f"{prefix}_fraction": 0.0,
                f"{prefix}_largest": 0.0,
                f"{prefix}_mean_size": 0.0,
                f"{prefix}_std_size": 0.0,
                f"{prefix}_threshold": threshold,
            }
        else:
            row = {
                f"{prefix}_n": int(puncta_sizes.size),
                f"{prefix}_total_voxels": float(puncta_sizes.sum()),
                f"{prefix}_fraction": float(puncta_sizes.sum() / cell_voxels),
                f"{prefix}_largest": float(puncta_sizes.max()),
                f"{prefix}_mean_size": float(puncta_sizes.mean()),
                f"{prefix}_std_size": float(puncta_sizes.std()),
                f"{prefix}_threshold": threshold,
            }

        feature_dict[int(label_id)] = row

    return feature_dict


def compute_z_distribution_features_fast(mask_data, red_channel, prefix="red_z_distribution"):
    feature_dict = {}

    labels = labels_from_mask(mask_data)

    if labels.size == 0:
        return feature_dict

    z_indices = np.arange(red_channel.shape[0], dtype=float)
    red_sum_by_label_z = []

    for z in range(red_channel.shape[0]):
        red_sum_z = ndi.sum(red_channel[z], labels=mask_data[z], index=labels)
        red_sum_by_label_z.append(red_sum_z)

    red_sum_by_label_z = np.vstack(red_sum_by_label_z)
    total_by_label = red_sum_by_label_z.sum(axis=0)

    for i, label_id in enumerate(labels):
        total = float(total_by_label[i])

        if total <= 0:
            row = {
                f"{prefix}_peak_z": np.nan,
                f"{prefix}_weighted_z": np.nan,
                f"{prefix}_z_spread": np.nan,
                f"{prefix}_peak_fraction": np.nan,
            }
        else:
            signal = red_sum_by_label_z[:, i]
            weighted_z = float(np.sum(z_indices * signal) / total)
            z_spread = float(np.sqrt(np.sum(((z_indices - weighted_z) ** 2) * signal) / total))
            peak_z = int(np.argmax(signal))

            row = {
                f"{prefix}_peak_z": peak_z,
                f"{prefix}_weighted_z": weighted_z,
                f"{prefix}_z_spread": z_spread,
                f"{prefix}_peak_fraction": float(signal[peak_z] / total),
            }

        feature_dict[int(label_id)] = row

    return feature_dict


def compute_glcm_features_for_labels_bbox(
    mask_data,
    intensity_image,
    levels=32,
    min_pixels=50,
    prefix="red_glcm",
):
    feature_dict = {}

    objects = ndi.find_objects(mask_data)

    for label_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue

        crop_mask = mask_data[slc] == label_id

        if int(crop_mask.sum()) < min_pixels:
            continue

        crop = intensity_image[slc].astype(np.float32)

        crop_mip = crop.max(axis=0)
        mask_mip = crop_mask.max(axis=0)

        pix = crop_mip[mask_mip]

        if pix.size < 20:
            continue

        lo, hi = np.percentile(pix, [1, 99])
        if hi <= lo:
            continue

        crop_scaled = np.clip((crop_mip - lo) / (hi - lo), 0, 1)
        crop_q = (crop_scaled * (levels - 1)).astype(np.uint8)
        crop_q[~mask_mip] = 0

        glcm = graycomatrix(
            crop_q,
            distances=[1, 2, 4],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=levels,
            symmetric=True,
            normed=True,
        )

        row = {}

        for prop in [
            "contrast",
            "dissimilarity",
            "homogeneity",
            "ASM",
            "energy",
            "correlation",
        ]:
            vals = graycoprops(glcm, prop)
            row[f"{prefix}_{prop}_mean"] = float(np.nanmean(vals))
            row[f"{prefix}_{prop}_std"] = float(np.nanstd(vals))

        feature_dict[int(label_id)] = row

    return feature_dict


# ---------------------------------------------------------------------------
# Tissue and DAPI positivity
# ---------------------------------------------------------------------------

def get_green_tissue_mask_and_area_um2(
    green_volume_zyx,
    xy_spacing,
    blur_sigma=GREEN_TISSUE_BLUR_SIGMA,
    threshold=GREEN_TISSUE_THRESHOLD,
):
    """
    Detect tissue from blurred green-channel MIP.

    The tissue is assumed to be either the whole FOV or one consistent piece.
    Therefore small objects are removed and only the largest connected component
    is retained.
    """
    green_mip = green_volume_zyx.max(axis=0).astype(np.float32)

    green_blurred = ndi.gaussian_filter(
        green_mip,
        sigma=blur_sigma,
    )

    tissue_mask = green_blurred > threshold

    if GREEN_TISSUE_OPENING_RADIUS > 0:
        tissue_mask = skimage.morphology.binary_opening(
            tissue_mask,
            skimage.morphology.disk(GREEN_TISSUE_OPENING_RADIUS),
        )

    if GREEN_TISSUE_CLOSING_RADIUS > 0:
        tissue_mask = skimage.morphology.binary_closing(
            tissue_mask,
            skimage.morphology.disk(GREEN_TISSUE_CLOSING_RADIUS),
        )

    if GREEN_TISSUE_FILL_HOLES:
        tissue_mask = ndi.binary_fill_holes(tissue_mask)

    tissue_mask = skimage.morphology.remove_small_objects(
        tissue_mask.astype(bool),
        min_size=GREEN_TISSUE_MIN_SIZE,
    )

    labeled = skimage.measure.label(tissue_mask)

    if labeled.max() > 0:
        props = skimage.measure.regionprops(labeled)
        largest_label = max(props, key=lambda p: p.area).label
        tissue_mask = labeled == largest_label
    else:
        tissue_mask = np.zeros_like(tissue_mask, dtype=bool)

    tissue_area_um2 = int(tissue_mask.sum()) * (xy_spacing ** 2)

    return tissue_mask, tissue_area_um2


def compute_dapi_positive_mask_3d(dapi_volume_zyx):
    dapi_norm = skimage.exposure.rescale_intensity(
        dapi_volume_zyx.astype(np.float32),
        out_range=(0, 65535),
    ).astype(np.uint16)

    thresh = skimage.filters.threshold_otsu(dapi_norm)
    dapi_mask = dapi_norm > thresh

    if DAPI_OPENING_RADIUS > 0:
        dapi_mask = skimage.morphology.binary_opening(
            dapi_mask,
            skimage.morphology.ball(DAPI_OPENING_RADIUS),
        )

    return dapi_mask


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def export_qc_crops(image_name, raw_channels_czyx, props_green, output_root, max_crops=100):
    crop_root = output_root / image_name
    crop_root.mkdir(parents=True, exist_ok=True)

    n_objects = len(props_green)
    if n_objects == 0:
        return

    rng = np.random.default_rng()
    selected_indices = rng.choice(n_objects, size=min(max_crops, n_objects), replace=False)

    for prop_index in selected_indices:
        prop = props_green[prop_index]
        z0, y0, x0, z1, y1, x1 = prop.bbox

        crop = raw_channels_czyx[:, z0:z1, y0:y1, x0:x1]
        crop_path = crop_root / f"{image_name}_label{prop.label:04d}_raw_crop.tif"

        tifffile.imwrite(
            str(crop_path),
            crop,
            metadata={"axes": "CZYX"},
        )


def save_overview_fast(
    image_name,
    dapi_channel,
    green_channel,
    red_channel,
    farred_channel,
    mask_data,
    tissue_mask_2d,
):
    scale = OVERLAY_SCALE

    mask_mip = mask_data.max(axis=0).astype(np.int32)

    mask_mip_s = skimage.transform.rescale(
        mask_mip,
        scale,
        order=0,
        anti_aliasing=False,
        preserve_range=True,
    ).astype(np.int32)

    tissue_s = skimage.transform.rescale(
        tissue_mask_2d.astype(np.uint8),
        scale,
        order=0,
        anti_aliasing=False,
        preserve_range=True,
    ).astype(bool)

    boundaries = skimage.segmentation.find_boundaries(mask_mip_s, mode="outer")

    channel_mips = [
        (dapi_channel.max(axis=0), "DAPI"),
        (green_channel.max(axis=0), "Green"),
        (red_channel.max(axis=0), "Red"),
        (farred_channel.max(axis=0), "Far-red"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)

    for ax, (mip, title) in zip(axes.ravel(), channel_mips):
        mip_s = skimage.transform.rescale(
            mip,
            scale,
            anti_aliasing=True,
            preserve_range=True,
        )

        finite = np.isfinite(mip_s)
        lo, hi = np.percentile(mip_s[finite], [1, 99]) if np.any(finite) else (0, 1)

        ax.imshow(mip_s, cmap="gray", vmin=lo, vmax=hi)
        ax.imshow(np.ma.masked_where(~boundaries, boundaries), cmap="hsv", alpha=1)
        ax.contour(tissue_s, levels=[0.5], colors=["lime"], linewidths=1.0)
        ax.set_title(title)
        ax.axis("off")

    fig.savefig(
        OVERVIEWS_ROOT / f"{image_name}_channel_mips_with_mask_outlines.png",
        dpi=OVERVIEW_DPI,
    )
    plt.close(fig)


def save_overlay(image_name, green_channel, red_channel, mask_data, props_df):
    invalid_labels = props_df.loc[~props_df["valid"], "label"].to_numpy()

    filtered_mask_data = mask_data.copy()
    filtered_mask_data[np.isin(filtered_mask_data, invalid_labels)] = 0

    green_mip = green_channel.max(axis=0)
    red_mip = red_channel.max(axis=0)
    mask_mip = filtered_mask_data.max(axis=0).astype(np.int32)

    green_mip_s = skimage.transform.rescale(
        green_mip,
        OVERLAY_SCALE,
        anti_aliasing=True,
        preserve_range=True,
    )

    red_mip_s = skimage.transform.rescale(
        red_mip,
        OVERLAY_SCALE,
        anti_aliasing=True,
        preserve_range=True,
    )

    mask_mip_s = skimage.transform.rescale(
        mask_mip,
        OVERLAY_SCALE,
        anti_aliasing=False,
        order=0,
        preserve_range=True,
    ).astype(np.int32)

    boundaries = skimage.segmentation.find_boundaries(mask_mip_s, mode="outer")

    rgb = np.zeros((*green_mip_s.shape, 3), dtype=np.uint8)
    rgb[..., 0] = _to_uint8(red_mip_s)
    rgb[..., 1] = _to_uint8(green_mip_s)
    rgb[boundaries] = [255, 255, 255]

    plt.imsave(OVERVIEWS_ROOT / f"{image_name}_mip_overlay.png", rgb)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"INPUT_ROOT does not exist: {INPUT_ROOT}")

    if not MASK_ROOT.exists():
        raise FileNotFoundError(f"MASK_ROOT does not exist: {MASK_ROOT}")

    mask_files = find_mask_files(MASK_ROOT, MASK_SUFFIX)

    if not mask_files:
        print("No eligible .tif mask files found.")
        return

    print(f"Found {len(mask_files)} eligible mask file(s).")

    records = []

    for i, mask_path in enumerate(mask_files, start=1):
        print(f"[{i}/{len(mask_files)}] Processing {mask_path}")

        mask_data = tifffile.imread(str(mask_path)).squeeze()
        print(f"  mask shape: {mask_data.shape}")

        if mask_data.ndim != 3:
            print(f"  WARNING: mask is not 3D, shape={mask_data.shape} — skipping.")
            continue

        mask_data = mask_data.astype(np.uint32, copy=False)

        mask_name = mask_path.stem
        image_name = mask_name.replace(MASK_SUFFIX, "")
        rel = mask_path.parent.relative_to(MASK_ROOT)
        image_path = INPUT_ROOT / rel / f"{image_name}.ims"

        if not image_path.exists():
            print(f"  WARNING: matching image not found: {image_path}")
            continue

        image_data, xy_spacing, z_spacing = read_ims_channels_bioio(
            image_path,
            channel_indices=[CH_DAPI, CH_GREEN, CH_RED, CH_FARRED],
            scene_index=BIOIO_SCENE_INDEX,
        )

        dapi_channel = image_data[CH_DAPI]
        green_channel = image_data[CH_GREEN]
        red_channel = image_data[CH_RED]
        farred_channel = image_data[CH_FARRED]

        print(f"  image shape CZYX: {image_data.shape}")
        print(f"  xy_spacing={xy_spacing:.4f} µm, z_spacing={z_spacing:.4f} µm")

        if mask_data.shape != green_channel.shape:
            image_data, scale_z, scale_y, scale_x = downscale_image_to_mask_shape(
                image_data,
                mask_data.shape,
            )

            z_spacing = z_spacing * scale_z
            xy_spacing = xy_spacing * scale_y

            dapi_channel = image_data[CH_DAPI]
            green_channel = image_data[CH_GREEN]
            red_channel = image_data[CH_RED]
            farred_channel = image_data[CH_FARRED]

            print(
                f"  Updated spacing after image downscaling: "
                f"xy_spacing={xy_spacing:.4f} µm, z_spacing={z_spacing:.4f} µm"
            )

        print("  Detecting tissue from green channel...")
        tissue_mask_2d, tissue_area_um2 = get_green_tissue_mask_and_area_um2(
            green_channel,
            xy_spacing,
        )

        dapi_positive_mask_3d = compute_dapi_positive_mask_3d(dapi_channel)

        print(f"  Tissue area: {tissue_area_um2:.1f} µm²")

        print("  Saving overview...")
        save_overview_fast(
            image_name,
            dapi_channel,
            green_channel,
            red_channel,
            farred_channel,
            mask_data,
            tissue_mask_2d,
        )

        spacing = (z_spacing, xy_spacing, xy_spacing)

        print("  Computing base regionprops...")
        props_dapi = skimage.measure.regionprops(mask_data, dapi_channel, spacing=spacing)
        props_green = skimage.measure.regionprops(mask_data, green_channel, spacing=spacing)
        props_red = skimage.measure.regionprops(mask_data, red_channel, spacing=spacing)
        props_farred = skimage.measure.regionprops(mask_data, farred_channel, spacing=spacing)

        labels = np.array([p.label for p in props_green], dtype=np.uint32)
        areas = np.array([p.area for p in props_green])
        valid = (areas >= MIN_AREA) & (areas <= MAX_AREA)

        dapi_overlap_labels = np.unique(mask_data[dapi_positive_mask_3d])
        dapi_overlap_labels = set(dapi_overlap_labels[dapi_overlap_labels > 0].astype(int))

        dapi_positive_overlap = [
            int(label_id) in dapi_overlap_labels
            for label_id in labels
        ]

        props_df_data = {
            "image_name": image_name,
            "label": labels,
            "area": areas,
            "num_pixels": [p.num_pixels for p in props_green],
            "dapi_positive_overlap": dapi_positive_overlap,
            "valid": valid,
            "tissue_area_um2": tissue_area_um2,
        }

        channel_props = {
            "dapi": props_dapi,
            "green": props_green,
            "red": props_red,
            "farred": props_farred,
        }

        for channel_name, props_list in channel_props.items():
            print(f"  Computing {channel_name} statistics...")

            feature_dict = compute_channel_statistics_from_regionprops(
                props_list,
                prefix=f"intensity_{channel_name}",
            )

            for label_id in labels:
                row = feature_dict.get(int(label_id), {})

                props_df_data.setdefault(f"intensity_mean_{channel_name}", []).append(
                    row.get(f"intensity_{channel_name}_mean", np.nan)
                )
                props_df_data.setdefault(f"intensity_std_{channel_name}", []).append(
                    row.get(f"intensity_{channel_name}_std", np.nan)
                )

                for percentile in INTENSITY_PERCENTILES:
                    props_df_data.setdefault(f"intensity_p{percentile}_{channel_name}", []).append(
                        row.get(f"intensity_{channel_name}_p{percentile}", np.nan)
                    )

        props_df = pd.DataFrame(props_df_data)

        feature_channels = {
            "red": red_channel,
            "farred": farred_channel,
        }

        for feature_channel_name, feature_channel_data in feature_channels.items():
            print(f"  Computing blurred {feature_channel_name} features...")
            for sigma_um in RED_BLUR_SIGMAS_UM:
                sigma_voxels = sigma_um_to_voxels(
                    sigma_um,
                    z_spacing,
                    xy_spacing,
                )

                channel_blurred = gaussian_filter(
                    feature_channel_data.astype(np.float32, copy=False),
                    sigma=sigma_voxels,
                )

                sigma_name = str(sigma_um).replace(".", "p")
                prefix = f"{feature_channel_name}_blur_{sigma_name}um"

                features = compute_channel_statistics_from_labels_bbox(
                    mask_data,
                    channel_blurred,
                    prefix,
                )

                props_df = add_feature_dict_to_props_df(props_df, features)

                del channel_blurred

            print(f"  Computing dilated {feature_channel_name} features fast...")
            features = compute_dilated_mask_features_fast(
                mask_data,
                feature_channel_data,
                radius=MASK_DILATION_RADIUS_VOXELS,
                prefix=f"{feature_channel_name}_dilated_r{MASK_DILATION_RADIUS_VOXELS}",
            )

            props_df = add_feature_dict_to_props_df(props_df, features)

            print(f"  Computing {feature_channel_name} puncta features fast...")
            features = compute_puncta_features_fast(
                mask_data,
                feature_channel_data,
                percentile=PUNCTA_PERCENTILE,
                prefix=f"{feature_channel_name}_puncta_p{PUNCTA_PERCENTILE}",
            )

            props_df = add_feature_dict_to_props_df(props_df, features)

            print(f"  Computing {feature_channel_name} puncta structure features...")
            features = compute_puncta_structure_features_bbox(
                mask_data,
                feature_channel_data,
                percentile=PUNCTA_PERCENTILE,
                min_puncta_voxels=PUNCTA_MIN_VOXELS,
                prefix=f"{feature_channel_name}_puncta_structure_p{PUNCTA_PERCENTILE}",
            )

            props_df = add_feature_dict_to_props_df(props_df, features)

            print(f"  Computing {feature_channel_name} Z-distribution features fast...")
            features = compute_z_distribution_features_fast(
                mask_data,
                feature_channel_data,
                prefix=f"{feature_channel_name}_z_distribution",
            )

            props_df = add_feature_dict_to_props_df(props_df, features)

            print(f"  Computing {feature_channel_name} GLCM features...")
            features = compute_glcm_features_for_labels_bbox(
                mask_data,
                feature_channel_data,
                levels=GLCM_LEVELS,
                min_pixels=GLCM_MIN_PIXELS,
                prefix=f"{feature_channel_name}_glcm",
            )

            props_df = add_feature_dict_to_props_df(props_df, features)

        props_df.to_csv(HISTOGRAM_ROOT / f"{image_name}_props.csv", index=False)

        print("  Exporting QC crops...")
        raw_channels_czyx = np.stack(
            [dapi_channel, green_channel, red_channel, farred_channel],
            axis=0,
        )

        export_qc_crops(
            image_name,
            raw_channels_czyx,
            props_green,
            QC_CROPS_ROOT,
        )

        valid_df = props_df[props_df["valid"]].copy()

        save_histogram(props_df["area"], "area (µm³)", f"{image_name}_area", HISTOGRAM_ROOT)
        save_histogram(props_df["num_pixels"], "num_pixels", f"{image_name}_num_pixels", HISTOGRAM_ROOT)

        for col in [
            "intensity_mean_green",
            "intensity_p95_green",
            "intensity_mean_red",
            "intensity_std_red",
            "intensity_p95_red",
            "intensity_mean_farred",
            "intensity_std_farred",
            "intensity_p95_farred",
            "red_blur_1p0um_p95",
            f"red_dilated_r{MASK_DILATION_RADIUS_VOXELS}_shell_p95",
            f"red_puncta_structure_p{PUNCTA_PERCENTILE}_fraction",
            "red_glcm_contrast_mean",
            "farred_blur_1p0um_p95",
            f"farred_dilated_r{MASK_DILATION_RADIUS_VOXELS}_shell_p95",
            f"farred_puncta_structure_p{PUNCTA_PERCENTILE}_fraction",
            "farred_glcm_contrast_mean",
        ]:
            if col in valid_df.columns:
                save_histogram(
                    valid_df[col],
                    col,
                    f"{image_name}_{col}",
                    HISTOGRAM_ROOT,
                )

        print("  Saving overlay...")
        save_overlay(image_name, green_channel, red_channel, mask_data, props_df)

        n_masks = len(props_green)
        n_valid_masks = int(valid.sum())
        density = n_valid_masks / tissue_area_um2 if tissue_area_um2 > 0 else float("nan")

        threshold_otsu = skimage.filters.threshold_otsu(areas) if areas.size else np.nan
        threshold_triangle = skimage.filters.threshold_triangle(areas) if areas.size else np.nan

        records.append({
            "image_name": image_name,
            "n_masks": n_masks,
            "n_valid_masks": n_valid_masks,
            "tissue_area_um2": tissue_area_um2,
            "n_valid_per_um2": density,
            "threshold_otsu": threshold_otsu,
            "threshold_triangle": threshold_triangle,
            "xy_spacing": xy_spacing,
            "z_spacing": z_spacing,
            "bioio_scene_index": BIOIO_SCENE_INDEX,
            "green_tissue_blur_sigma": GREEN_TISSUE_BLUR_SIGMA,
            "green_tissue_threshold": GREEN_TISSUE_THRESHOLD,
        })

        del image_data
        del raw_channels_czyx

    df = pd.DataFrame(records)
    out_csv = HISTOGRAM_ROOT / "summary.csv"
    df.to_csv(out_csv, index=False)

    print(f"\nSummary saved to {out_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
