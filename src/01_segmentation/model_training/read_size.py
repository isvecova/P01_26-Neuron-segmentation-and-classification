from pathlib import Path
from typing import List

import numpy as np
from bioio import BioImage
from cellpose import io, transforms
import csv


# Edit these paths directly before running from the IDE.
INPUT_ROOT = Path(r"D:\OneDrive - IEM\000_inbox\260604_Kralova_python_troubleshooting\cellpose + far-red channel\data")
OUTPUT_ROOT = INPUT_ROOT / ".."


def find_ims_files(root: Path) -> List[Path]:
    return sorted(path for path in root.rglob("*.ims") if path.is_file())


def main():
    image_names = find_ims_files(INPUT_ROOT)
    if not image_names:
        raise ValueError(f"No .ims files found under {INPUT_ROOT}")

    for name in image_names:
        print(f"Processing {name}...")
        image = BioImage(str(name))

        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        csv_path = OUTPUT_ROOT / "metadata.csv"
        write_header = not csv_path.exists()

        def _lookup(obj, *keys):
            cur = obj
            for k in keys:
                try:
                    if isinstance(k, int):
                        cur = cur[k]
                    else:
                        cur = getattr(cur, k)
                except Exception:
                    try:
                        cur = cur[k]
                    except Exception:
                        try:
                            cur = cur.get(k)
                        except Exception:
                            return None
            return cur

        with open(csv_path, "a", newline="", encoding="utf8") as fh:
            writer = csv.writer(fh)
            if write_header:
                writer.writerow(["relative_path", "primary_subfolder", "x", "y", "z"])

            try:
                rel = name.relative_to(INPUT_ROOT)
            except Exception:
                rel = Path(name.name)
            primary = rel.parts[0] if len(rel.parts) >= 2 else ""

            px = _lookup(image.metadata, "images", 0, "pixels", "physical_size_x")
            py = _lookup(image.metadata, "images", 0, "pixels", "physical_size_y")
            pz = _lookup(image.metadata, "images", 0, "pixels", "physical_size_z")

            writer.writerow([str(rel), primary, px, py, pz])



if __name__ == '__main__':
    main()
