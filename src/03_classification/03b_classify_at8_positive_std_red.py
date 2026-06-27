#!/usr/bin/env python3
"""
Global AT8 classification using red intensity standard deviation.

Pools intensity_std_red values across all per-image props CSVs,
computes a global threshold using triangle thresholding (robust to
imbalanced classes), and reports the fraction of AT8-positive cells per image.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.filters
from pathlib import Path

# =========================
# CONFIGURATION
# =========================
CSV_ROOT  = Path(r"L:\0_Service\Sarah\Croatia project\histograms")  # folder containing *_props.csv files
CSV_GLOB  = "*_props.csv"
OUT_DIR   = CSV_ROOT  # summary CSV and histogram saved here
VALUE_COL = "intensity_std_red"

# --- Optional cell filtering before classification ---
# Set to True to apply pre-filtering; False to classify all cells.
FILTER_BY_VALID = True   # Use only cells marked as 'valid=True' in the CSV
FILTER_BY_AREA = False   # Use only cells within [MIN_AREA, MAX_AREA]

# Area filter thresholds (only used if FILTER_BY_AREA=True)
MIN_AREA = 300
MAX_AREA = 4000


def main() -> None:
    # --- 1. Discover and validate input CSV files ---
    csv_files = sorted(CSV_ROOT.glob(CSV_GLOB))
    if not csv_files:
        raise FileNotFoundError(f"No files matching '{CSV_GLOB}' found in {CSV_ROOT}")

    print(f"Found {len(csv_files)} CSV file(s).")
    print(f"Target column: {VALUE_COL}")
    print(f"Filtering: FILTER_BY_VALID={FILTER_BY_VALID}, FILTER_BY_AREA={FILTER_BY_AREA}")
    if FILTER_BY_AREA:
        print(f"  Area range: [{MIN_AREA}, {MAX_AREA}]")
    print()

    # --- 2. Load all CSVs and apply optional filtering ---
    dfs: dict[str, pd.DataFrame] = {}
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        if VALUE_COL not in df.columns:
            print(f"  WARNING: '{VALUE_COL}' not found in {csv_path.name}, skipping.")
            continue

        # Apply cell filtering if enabled
        original_count = len(df)
        if FILTER_BY_VALID and 'valid' in df.columns:
            df = df[df['valid'] == True].copy()
        if FILTER_BY_AREA and 'area' in df.columns:
            df = df[(df['area'] >= MIN_AREA) & (df['area'] <= MAX_AREA)].copy()
        
        filtered_count = len(df)
        if filtered_count < original_count:
            print(f"  {csv_path.name}: filtered {original_count} -> {filtered_count} cells")
        
        dfs[csv_path.stem] = df

    if not dfs:
        raise ValueError(f"No usable CSV files found (column '{VALUE_COL}' missing in all).")

    # --- 3. Pool values across all images for global threshold calculation ---
    pooled = np.concatenate([df[VALUE_COL].dropna().to_numpy() for df in dfs.values()])
    pooled = pooled[np.isfinite(pooled)]
    print(f"\nPooled {len(pooled)} cell measurements across all images.")

    # --- 4. Compute global threshold on the pooled distribution ---
    # Triangle thresholding is suitable for unimodal/skewed distributions
    # where the positive class is a small minority.
    threshold = skimage.filters.threshold_triangle(pooled)
    print(f"Global triangle threshold on pooled {VALUE_COL}: {threshold:.4f}\n")

    # --- 5. Generate pooled histogram with threshold line ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.hist(pooled, bins=100, edgecolor="black", alpha=0.8, label="All cells")
    plt.axvline(threshold, color="red", linewidth=1.5, label=f"Triangle threshold = {threshold:.2f}")
    plt.xlabel(VALUE_COL)
    plt.ylabel("Count")
    plt.yscale("log")  # log scale to better visualize the positive tail
    plt.title(f"Pooled {VALUE_COL} — all images")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"pooled_{VALUE_COL}_histogram.png", dpi=200)
    plt.close()

    # --- 6. Classify cells per image using the global threshold ---
    records = []
    for name, df in dfs.items():
        values = df[VALUE_COL].dropna().to_numpy()
        values = values[np.isfinite(values)]
        n_total    = len(values)
        n_positive = int(np.sum(values > threshold))
        ratio      = n_positive / n_total if n_total > 0 else float("nan")
        records.append({
            "image":      name,
            "n_total":    n_total,
            "n_positive": n_positive,
            "ratio_positive": ratio,
        })
        print(f"  {name}: {n_positive}/{n_total} positive ({ratio:.1%})")

    # --- 7. Export per-image summary to CSV ---
    summary_df = pd.DataFrame(records)
    out_csv = OUT_DIR / "AT8_std_positive_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"\nSummary saved to {out_csv}")
    print(f"Threshold used: {threshold:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
