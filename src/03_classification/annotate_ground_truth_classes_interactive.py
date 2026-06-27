# Script for manual annotation of ground truth classes for cells in 3D images.
# An interactive napari viewer is launched, where the user can place points inside segmented cells.
# Points are saved to a CSV file, along with the label of the cell they are placed in, after pressing the "Save annotations" button.
# User can switch between images using the dropdown menu, and the script will load existing annotations if they exist.

from pathlib import Path
import numpy as np
import pandas as pd
import napari
import tifffile as tiff
from bioio import BioImage
from skimage.transform import resize


# -----------------------------
# User settings
# -----------------------------

# mask_path = Path(r"L:\Algernon\530\Sarah\segmentation_output_pseudo3d_260611\motor cortex\64U-J9910_CHST11KO_MC_AT8_delta4_first_2026-04-23_12.12.05_FusionStitcher_mask_3d.tif")
# image_path = Path(r"N:\01_scientific_data\Sarah_Kralova_brain_slice_data\cellpose + far-red channel\motor cortex\64U-J9910_CHST11KO_MC_AT8_delta4_first_2026-04-23_12.12.05_FusionStitcher.ims")


# mask_path = Path(r"L:\Algernon\530\Sarah\segmentation_output_pseudo3d_260611\motor cortex\1.3_P301S+PLA_MC_AT8_E2_first bottom_2026-04-22_14.10.39_FusionStitcher_mask_3d.tif")
# image_path = Path(r"N:\01_scientific_data\Sarah_Kralova_brain_slice_data\cellpose + far-red channel\motor cortex\1.3_P301S+PLA_MC_AT8_E2_first bottom_2026-04-22_14.10.39_FusionStitcher.ims")

image_name = "1.3_P301S+PLA_BS_AT8_E2_first bottom_2026-04-22_14.32.11_FusionStitcher"
# image_name = "64U-J9910_CHST11KO_BS_AT8_delta4_first_2026-04-23_12.32.20_FusionStitcher"
region = "brainstem"
mask_path = f"L:/Algernon/530/Sarah/segmentation_output_pseudo3d_260611/{region}/{image_name}_mask_3d.tif"
image_path = f"N:/01_scientific_data/Sarah_Kralova_brain_slice_data/cellpose + far-red channel/{region}/{image_name}.ims"

output_csv = Path(f"../../classification/annotations/cell_point_annotations_{image_name}.csv")

green_channel = 1
red_channel = 2
timepoint = 0

NON_BINNED_STEP = 0.152

MASK_ROOT = Path(r"L:\Algernon\530\Sarah\segmentation_output_pseudo3d_260611")
root_folder = Path(
    r"N:\01_scientific_data\Sarah_Kralova_brain_slice_data\cellpose + far-red channel"
)

image_paths = sorted(root_folder.rglob("*.ims"))

# Names for the dropdown
image_files = {}
regions = {}

for ims in sorted(root_folder.rglob("*.ims")):
    region = ims.parent.name
    mask = MASK_ROOT / region / f"{ims.stem}_mask_3d.tif"

    if mask.exists():
        image_files[ims.stem] = ims
        regions[ims.stem] = region

image_names = sorted(image_files.keys())


print(f"Found {len(image_names)} images.")

# -----------------------------
# Load image as 3D, no MIP
# -----------------------------




# -----------------------------
# Helper functions
# -----------------------------

def load_image_and_mask(image_name):
    image_path = image_files[image_name]
    region = regions[image_name]

    mask_path = (
        MASK_ROOT
        / region
        / f"{image_name}_mask_3d.tif"
    )

    img = BioImage(image_path)
    if img.scenes:
        x_step = img.metadata.images[0].pixels.physical_size_x
        if x_step is not None and np.abs(x_step - NON_BINNED_STEP) < 0.1:
            scene_index = 2
            print(f"  Detected non-binned image, using scene_index={scene_index}")
        else: 
            scene_index = 1
            print(f"  Detected binned image, using scene_index={scene_index}")
        img.set_scene(img.scenes[scene_index])
        
    # CZYX = channels, z, y, x
    # If your data has time, T=0 selects one timepoint.
    data = img.get_image_data("CZYX", T=timepoint)

    green = data[green_channel]
    red = data[red_channel]

    labels = tiff.imread(mask_path)

    # Make sure labels are ZYX
    labels = np.asarray(labels)

    labels = resize(
        labels,
        output_shape=red.shape,   # shape of your low-resolution image
        order=0,                  # nearest neighbour
        preserve_range=True,
        anti_aliasing=False,
    ).astype(labels.dtype)

    if labels.ndim != 3:
        raise ValueError(f"Expected a 3D mask in ZYX order, got shape {labels.shape}")

    if red.shape != labels.shape:
        raise ValueError(
            f"Image and mask shapes do not match.\n"
            f"Red shape: {red.shape}\n"
            f"Mask shape: {labels.shape}"
        )

    labels = labels.astype(np.uint32)

    return green, red, labels

    
def get_output_csv(image_name):
    output_dir = Path("../../classification/annotations")
    output_dir.mkdir(exist_ok=True)
    return output_dir / f"cell_point_annotations_{image_name}.csv"


def load_existing_annotations(image_name):

    output_csv = get_output_csv(image_name)

    if not output_csv.exists():
        print(f"No existing annotation file found: {output_csv}")
        return np.empty((0, 3)), np.empty((0, 3))

    df = pd.read_csv(output_csv)

    required = {"image", "annotation", "z", "y", "x"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Existing annotation file is missing columns: {missing}")

    df = df[df["image"] == image_name].copy()

    positive = df[df["annotation"] == 1][["z", "y", "x"]].to_numpy(float)
    negative = df[df["annotation"] == 0][["z", "y", "x"]].to_numpy(float)

    print(
        f"Loaded existing annotations from {output_csv}: "
        f"{len(positive)} positive, {len(negative)} negative"
    )

    return positive, negative


def point_to_label(point, labels):
    """
    Convert napari point coordinate z,y,x to integer mask label.
    """
    z, y, x = np.round(point).astype(int)

    if (
        z < 0 or z >= labels.shape[0] or
        y < 0 or y >= labels.shape[1] or
        x < 0 or x >= labels.shape[2]
    ):
        return 0

    return int(labels[z, y, x])


def collect_points(points_layer, annotation_value, labels, image_name):
    rows = []

    for i, point in enumerate(points_layer.data):
        label_id = point_to_label(point, labels)

        rows.append({
            "image": image_name,
            "point_index": i,
            "annotation": annotation_value,
            "label": label_id,
            "z": float(point[0]),
            "y": float(point[1]),
            "x": float(point[2]),
        })

    return rows


def switch_image(new_image_name):
    global image_name, output_csv, labels

    image_name = new_image_name
    output_csv = get_output_csv(image_name)

    green, red, labels = load_image_and_mask(image_name)
    existing_positive, existing_negative = load_existing_annotations(image_name)

    green_layer.data = green
    red_layer.data = red
    labels_layer.data = labels

    positive_points.data = existing_positive
    negative_points.data = existing_negative

    print(f"Loaded {image_name}")
    print(f"Output CSV: {output_csv}")


def save_annotations(image_name):
    if not positive_points.data.size and not negative_points.data.size:
        print("No points to save.")
        return
    
    rows = []

    rows.extend(collect_points(positive_points, 1, labels, image_name))
    rows.extend(collect_points(negative_points, 0, labels, image_name))

    df = pd.DataFrame(rows)

    # Remove points not placed inside a labelled cell
    df_inside = df[df["label"] > 0].copy()

    if df_inside.empty:
        print("No valid points inside labelled cells. Existing CSV was not overwritten.")
        return

    # Detect conflicts: same cell marked both positive and negative
    conflicts = (
        df_inside
        .groupby("label")["annotation"]
        .nunique()
    )
    conflict_labels = conflicts[conflicts > 1].index.tolist()

    if conflict_labels:
        print("WARNING: conflicting annotations for labels:")
        print(conflict_labels)

    df_inside.to_csv(output_csv, index=False)

    print(f"Saved {len(df_inside)} point annotations to {output_csv}")
    print(f"Ignored {len(df) - len(df_inside)} points outside labelled cells.")


# -----------------------------
# Napari viewer
# -----------------------------

if not image_names:
    raise RuntimeError("No images with matching masks were found.")

output_csv = get_output_csv(image_name)

green, red, labels = load_image_and_mask(image_name)
existing_positive, existing_negative = load_existing_annotations(image_name)


viewer = napari.Viewer(ndisplay=3)

# Load first image before creating viewer layers
green, red, labels = load_image_and_mask(image_name)

green_layer = viewer.add_image(
    green,
    name="green",
    colormap="green",
    blending="additive",
)

red_layer = viewer.add_image(
    red,
    name="red",
    colormap="red",
    blending="additive",
)

labels_layer = viewer.add_labels(
    labels,
    name="cell masks",
    opacity=0.35,
)

existing_positive, existing_negative = load_existing_annotations(image_name)

positive_points = viewer.add_points(
    existing_positive,
    name="positive points",
    ndim=3,
    size=8,
    face_color="red",
    border_color="white",
)

negative_points = viewer.add_points(
    existing_negative,
    name="negative points",
    ndim=3,
    size=8,
    face_color="blue",
    border_color="white",
)

positive_points.mode = "add"


# -----------------------------
# Keyboard shortcuts
# -----------------------------

from magicgui import magicgui

@magicgui(
    image_choice={"choices": image_names},
    call_button="Load selected image",
)
def image_switch_widget(image_choice=image_name):
    switch_image(image_choice)


viewer.window.add_dock_widget(
    image_switch_widget,
    area="right",
    name="Switch image",
)


@magicgui(
    call_button="Save annotations",
)
def save_widget():
    save_annotations(image_name)



viewer.window.add_dock_widget(
    save_widget,
    area="right",
    name="Save annotations",
)


print("""
Place points directly inside segmented cells.
The script reads the cell label under each point in 3D.
""")

napari.run()