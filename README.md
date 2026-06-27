# Update: June 2026
Second round of images was done with far-red channel. Moreover, the classification of red channel based on mean intensity only was no longer sufficient. 
Therefore, the scripts were adapted to extract more features and include training of a classifier. 

## Ground truth annotation and classifier training
### `.src\03_classification\annotate_ground_truth_classes_interactive.py`
What it does: 
- Finds a list of `.ims` in predefined root folder. 
- Loads one of the images and displays it in napari. 
- Allows for point annotation of positive/negative cells (by placing points into the two Points layers, so that they overlap with the annotated label). 
- After pressing Save annotations in the widget, the annotations are stored in the `.classification\annotations` folder in `.csv` format. 
- User can switch between different images in napari. If the loaded image already has annotations present, they are automatically loaded. 

Output: 
- `.csv` files with annotations (contain information about image name, class, label it was assigned to and point position)
  - can be subsequently used for model training

Prerequisites:
- Python packages: `bioio`, `napari`, `numpy`, `pandas`, `skimage`, `tifffile`.
- Input data accessible at configured paths.


### `.src\03_classification\classification_model_training.ipynb`
Jupyter notebook for interactive data loading, model training and result visualization. 
Prerequisites:
- Python packages: `bioio`, `napari`, `numpy`, `pandas`, `skimage`, `sklearn`,`tifffile`.
- Input data accessible at configured paths.

---

# AT8 Neuron segmentation workflow

## 1) Goal of the analysis
The goal of the analysis is to measure the number of AT8 positive cells in different conditions.
This workflow segments cells in 3D confocal microscopy `.ims` data from the NeuN channel using Cellpose, filters masks by size (possibly also by mean intensity of the NeuN channel), and classifies cells as AT8-positive/negative using global threshold computed across all images.

### Expected dataset
**Filetype:**
- `.ims` files (Imaris format for 3D confocal microscopy data)

**Channels:**
- Minimum 2 channels required:
  - **Channel 1** (index 1 in script): NeuN marker (green channel) - used for 3D cell segmentation
  - **Channel 2** (index 2 in script): AT8 marker (red channel) - used for AT8-positive classification

**Data structure:**
- 3D confocal microscopy Z-stacks
- Anisotropic voxel spacing typical (Z-spacing usually larger than XY-spacing)
- Multi-image datasets organized in nested folder structure supported
- Files containing "Overview" in the name are automatically excluded from processing

**Size:**
- No strict size limitations, but images are downscaled 4× in XY during segmentation for performance
- Typical dataset: multiple `.ims` files with cca 10 Z-planes per image

## 2) Scripts
Use scripts in this order:
1. `01_segment_ims_cellpose_sam_3d.py`: runs 3D Cellpose-SAM segmentation on `.ims` 3D data and writes `_mask_3d.tif` label masks.
2. `mask_exploration.ipynb` (optional): after segmentation, use this notebook to test filtering parameters and visually inspect masks vs signal before batch measurement. It loads downscaled image and mask data for faster interactive exploration.
3. `02_measure_cells_and_generate_qc.py`: measures per-cell morphology and intensities from masks and raw images, then exports CSV and QC plots.
4. `03a_classify_at8_positive_mean_red.py`: pools `intensity_mean_red` values, computes a global threshold, and reports AT8-positive ratios.
5. `03b_classify_at8_positive_std_red.py`: alternative classifier using `intensity_std_red` to compute AT8-positive ratios.

## 3) Script-by-script details

### `01_segment_ims_cellpose_sam_3d.py`
What it does:
- Recursively finds `.ims` files under `INPUT_ROOT` (except names containing `Overview`).
- Extracts one channel (`CHANNEL_INDEX = 1`), applies per-slice median filtering.
- Runs 3D Cellpose-SAM segmentation on downscaled XY volumes.
- Resizes masks back to original size and saves `_mask_3d.tif`.
- Skips files where output mask already exists.

Inputs:
- Root folder with `.ims` files: `INPUT_ROOT`.
- Configuration parameters in script:
  - `CHANNEL_INDEX`, `CELLPROB_THRESHOLD`, `USE_GPU`, `MODEL_TYPE`, `SKIP_NAME_TOKEN`.

Outputs:
- Segmentation masks at:
  - `OUTPUT_ROOT/<relative_subfolder>/<image_stem>_mask_3d.tif`

Prerequisites:
- Python packages: `bioio`, `cellpose`, `numpy`, `skimage`, `tifffile`.
- GPU recommended for performance (`USE_GPU=True`).
- Input data accessible at configured `L:` paths.

### `02_measure_cells_and_generate_qc.py`
What it does:
- Finds mask files (`*_mask_3d*.tif`) under `MASK_ROOT`.
- Locates matching `.ims` file using mask filename + relative subfolder logic.
- Loads channels 1 and 2 from image (NeuN/green, AT8/red).
- Computes regionprops for each label with physical spacing.
- Applies size validity filter (`min_area`, `max_area`).
- Saves per-cell CSVs, parameter histograms, and 2D overlay QC images.
- Produces a cross-image `summary.csv`.

Inputs:
- Mask root: `MASK_ROOT`.
- Raw image root: `INPUT_ROOT`.
- Thresholds and path settings in script.

Outputs:
- Per-image measurements CSV:
  - `HISTOGRAM_ROOT/<image_name>_props.csv`
- Per-image histograms (PNG):
  - area, voxel count, mean green, mean red, std red.
- Per-image overlay QC PNG:
  - `OVERVIEWS_ROOT/<image_name>_mip_overlay.png`
- Combined summary:
  - `HISTOGRAM_ROOT/summary.csv`

Prerequisites:
- Python packages: `bioio`, `numpy`, `pandas`, `matplotlib`, `skimage`, `tifffile`.
- Masks from step 1 available with expected naming convention.

### `03a_classify_at8_positive_mean_red.py`
What it does:
- Loads all `*_props.csv` files.
- Applies optional filtering (see below).
- Pools `intensity_mean_red` across all cells from all images.
- Computes one global triangle threshold.
- Classifies each cell as positive if `intensity_mean_red > threshold`.
- Reports and saves per-image positive ratios.

Inputs:
- Folder of per-image props CSVs: `CSV_ROOT`.
- Column: `VALUE_COL = "intensity_mean_red"`.

Optional filtering (configured at top of script):
- `FILTER_BY_VALID = True/False`: if True, uses only cells marked as `valid=True` in the CSV (size-filtered cells from step 2).
- `FILTER_BY_AREA = True/False`: if True, applies additional area filter using `MIN_AREA` and `MAX_AREA` thresholds.
- The script prints filtering status and per-file statistics at runtime for transparency.

Outputs:
- Pooled histogram with threshold:
  - `pooled_intensity_mean_red_histogram.png`
- Per-image summary CSV:
  - `AT8_positive_summary.csv`

Prerequisites:
- Python packages: `numpy`, `pandas`, `matplotlib`, `skimage`.
- Props CSVs from step 2.

### `03b_classify_at8_positive_std_red.py`
What it does:
- Same pooled-threshold classification pattern as previous script.
- Uses `intensity_std_red` as the classification feature (alternative metric).
- Includes the same optional filtering options.

Inputs:
- `*_props.csv` files in `CSV_ROOT`.
- Column: `VALUE_COL = "intensity_std_red"`.

Optional filtering (configured at top of script):
- `FILTER_BY_VALID = True/False`: if True, uses only cells marked as `valid=True` in the CSV.
- `FILTER_BY_AREA = True/False`: if True, applies area filter using `MIN_AREA` and `MAX_AREA` thresholds.
- Same filtering transparency as script 03a.

Outputs:
- Pooled histogram:
  - `pooled_intensity_std_red_histogram.png`
- Per-image summary CSV:
  - `AT8_std_positive_summary.csv`

Prerequisites:
- Same as `03a_classify_at8_positive_mean_red.py`.
- Props CSVs from step 2.

## 4) Rationale behind parameter selection

### Segmentation parameters (`01_segment_ims_cellpose_sam_3d.py`)
- `CELLPROB_THRESHOLD = -5.0`: Low cellprop threshold value selected to include as much of the mask as possible and to avoid capturing only the brighter part of the cell (some cells contain more green signal in the nucleus and less in the cytoplasm).
- `CHANNEL_INDEX = 1`: Green channel, which corresponds to NeuN, is on position 2.
- XY downscaling factor (4×): 4x downscaling in XY preserves the necessary level of detail while speeding the analysis. 
- `ANISOTROPY = 1`: Even though the real anisotropy of the downscaled images is 3 (z-spacing is 3x the downscaled xy), using anisotropy of 3 did not provide significant improvement, but it slowed the processing quite a lot -> we use 1 instead.
- Both Cellpose3 and 2D CellposeSAM were tested, but they created substantially more artifacts -> 3D CellposeSAM provided the best segmentation.

### Size filtering (`02_measure_cells_and_generate_qc.py`)

### Cell filtering in classification scripts (`03a` and `03b`)
- `FILTER_BY_VALID`: When enabled, only cells that passed size filtering in step 2 are included in the threshold calculation and classification. Recommended to enable this to exclude artifacts.
- `FILTER_BY_AREA`: When enabled, applies an additional explicit area filter. Useful for testing alternative thresholds without re-running step 2.
- Both filters can be combined or used independently.
- `min_area = 300`: Removes small debree and segmentation artifacts, as well as masks existing only in one or two planes.
- `max_area = 4000`: 

### Classification method
- Triangle thresholding: Triangle thresholding should be suitable for dividing data into two disproportionate classes. If needed, other options can be tested.
- Mean red vs std red: Both mean and std values seem to show difference between treatment and control.

## 5) How to use

### Environment setup

#### Required packages
All scripts require these Python packages:
- `numpy` - numerical computing
- `pandas` - data manipulation and CSV handling
- `matplotlib` - plotting and visualization
- `scikit-image` - image processing utilities
- `bioio` - microscopy image I/O (supports .ims files)
- `tifffile` - TIFF file reading/writing
- `cellpose` - deep learning segmentation (includes PyTorch)

For the interactive exploration notebook (`mask_exploration_intensity.ipynb`), you also need:
- `napari` - 3D image viewer

#### Installing miniforge and creating the environment
(Already available on Algernon computer as `cellposeProcessing` environment.)

1. **Download and install miniforge** (recommended over Anaconda/Miniconda because of licensing):
   - Visit https://conda-forge.org/download/
   - Download the Windows installer: `Miniforge3-Windows-x86_64.exe`
   - Run the installer and follow the prompts

2. **Open a new Miniforge Prompt** and verify the installation:
   ```powershell
   conda --version
   ```

3. **Create a dedicated conda environment** for this project:
   ```powershell
   conda create -n at8_analysis python=3.10 -y
   ```

4. **Activate the environment**:
   ```powershell
   conda activate at8_analysis
   ```

5. **Install required packages**:
   ```powershell
   # Core scientific packages and BioIO for import
   pip install bioio bioio-bioformats numpy pandas matplotlib scikit-image tifffile
   
   # Cellpose for segmentation
   pip install cellpose

   # For GPU support in Cellpose, follow the official instructions available her: https://pypi.org/project/cellpose/
   
   # Optional: Napari for interactive visualization
   pip install napari[all]
   ```

### Running the analysis

From the `scripts/` directory with the conda environment activated:

```powershell
conda activate at8_analysis
python .\01_segment_ims_cellpose_sam_3d.py
python .\02_measure_cells_and_generate_qc.py
python .\03a_classify_at8_positive_mean_red.py
python .\03b_classify_at8_positive_std_red.py
```

For the interactive notebook, use Jupyter (possible in VS Code). 

**Recommended pre-run checks:**
- Confirm `INPUT_ROOT` and related output paths exist and are writable.
- Confirm expected channels match your `.ims` layout.
- Edit path constants at the top of each script to match your data location.
- For GPU acceleration in segmentation, ensure you have a CUDA-compatible GPU and drivers installed: https://pypi.org/project/cellpose/

## 6) Outputs
- `*_mask_3d.tif`: 3D label masks from Cellpose-SAM segmentation.
- `<image>_props.csv`: one row per segmented object with size/intensity metrics.
- `*_histogram.png`: distribution QC plots of measured parameters.
- `<image>_mip_overlay.png`: quick visual QC of segmentation boundaries over signal MIP.
- `summary.csv`: per-image processing summary (counts, thresholds, spacing metadata).
- `AT8_positive_summary.csv`: per-image AT8-positive ratio using mean red intensity threshold.
- `AT8_std_positive_summary.csv`: per-image AT8-positive ratio using red intensity standard deviation threshold.

## 7) Time breakdown


| Activity | Estimated duration (hours) | 
|---|---|
| Testing segmentation using classical image processing | 1.5 | 
| Cellpose testing + batch segmentation script | 2 |
| Segmentation visualization + visualization script | 1 |
| Per-cell measurements, identification of suitable filtering + batch processing script | 2 |
| Cell classification by AT8 presence + batch measurement script | 1 |
| Documentation, troubleshooting | 2 |
| Troubleshooting updated script | 3 |
| Clustering, preparation of classification scripts | 6 |
| Manual annotation, testing | 2 |
| Optimization, cleanup, documentation | 3 |
| | |
