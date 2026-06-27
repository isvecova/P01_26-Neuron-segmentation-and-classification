from pathlib import Path
from typing import List

import numpy as np
from bioio import BioImage
from cellpose import io, transforms


# Edit these paths directly before running from the IDE.
INPUT_ROOT = Path(r"D:\OneDrive - IEM\000_inbox\260604_Kralova_python_troubleshooting\cellpose + far-red channel")
OUTPUT_ROOT = INPUT_ROOT / "train"

# BioImage scene index is zero-based, so scene 2 means the third scene.
SCENE_INDEX = 2

# Three random XY crops per .ims file.
N_RANDOM_CROPS = 3
CROP_SIZE = 100

# Keep the existing Cellpose training-image generation behavior.
NIMG_PER_TIF = 10
SHARPNEN_RADIUS = 0.0
TILE_NORM = 0
CHANNEL_AXIS = 0
Z_AXIS = 1
ANISOTROPY = 1.0


def find_ims_files(root: Path) -> List[Path]:
    return sorted(path for path in root.rglob("*.ims") if path.is_file())


def load_scene_image(path: Path, scene_index: int) -> np.ndarray:
    image = BioImage(str(path))
    if scene_index >= len(image.scenes):
        raise ValueError(f"{path} has only {len(image.scenes)} scene(s); cannot load scene {scene_index}.")

    image.set_scene(image.scenes[scene_index])
    arr = image.get_image_data().squeeze()

    # Cellpose expects a 3D or 4D array after the axes are normalized.
    if arr.ndim == 3:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 4:
        raise ValueError(f"Expected a 3D/4D array from {path}, got shape {arr.shape}.")

    return transforms.convert_image(arr, channel_axis=CHANNEL_AXIS, z_axis=Z_AXIS, do_3D=True)


def random_crop_start(length: int, crop_size: int, rng: np.random.Generator) -> int:
    if length <= crop_size:
        return 0
    return int(rng.integers(0, length - crop_size + 1))


def main():
    image_names = find_ims_files(INPUT_ROOT)
    if not image_names:
        raise ValueError(f"No .ims files found under {INPUT_ROOT}")

    rng = np.random.default_rng(0)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    pm = [(0, 1, 2, 3), (2, 0, 1, 3), (1, 0, 2, 3)]
    npm = ["YX", "ZY", "ZX"]

    for name in image_names:
        rel_name = name.relative_to(INPUT_ROOT).with_suffix("").as_posix().replace("/", "__")
        img0 = load_scene_image(name, SCENE_INDEX)

        if img0.shape[1] < CROP_SIZE or img0.shape[2] < CROP_SIZE:
            print(f"Skipping {name} because scene {SCENE_INDEX} is smaller than {CROP_SIZE}x{CROP_SIZE} in XY: {img0.shape}")
            continue

        for crop_index in range(N_RANDOM_CROPS):
            y0 = random_crop_start(img0.shape[1], CROP_SIZE, rng)
            x0 = random_crop_start(img0.shape[2], CROP_SIZE, rng)
            crop0 = img0[:, y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE, ...]

            for p in range(3):
                img = crop0.transpose(pm[p]).copy()
                print(rel_name, npm[p], img[0].shape)
                ly_max = max(0, img.shape[1] - CROP_SIZE)
                lx_max = max(0, img.shape[2] - CROP_SIZE)
                imgs = img[rng.permutation(img.shape[0])[:NIMG_PER_TIF]]
                if ANISOTROPY > 1.0 and p > 0:
                    imgs = transforms.resize_image(imgs, Ly=int(ANISOTROPY * img.shape[1]), Lx=img.shape[2])

                for k, plane_img in enumerate(imgs):
                    if TILE_NORM:
                        plane_img = transforms.normalize99_tile(plane_img, blocksize=TILE_NORM)
                    if SHARPNEN_RADIUS:
                        plane_img = transforms.smooth_sharpen_img(plane_img, sharpen_radius=SHARPNEN_RADIUS)

                    ly = 0 if ly_max == 0 else int(rng.integers(0, ly_max + 1))
                    lx = 0 if lx_max == 0 else int(rng.integers(0, lx_max + 1))
                    out_name = f"{rel_name}_crop{crop_index + 1}_{npm[p]}_{k}.tif"
                    io.imsave(
                        str(OUTPUT_ROOT / out_name),
                        plane_img[ly:ly + CROP_SIZE, lx:lx + CROP_SIZE].squeeze(),
                    )


if __name__ == '__main__':
    main()
