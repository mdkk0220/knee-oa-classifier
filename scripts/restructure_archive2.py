# scripts/restructure_archive2.py

import os
from pathlib import Path
import shutil

RAW_ROOT = Path("data/raw/archive2")
OUT_ROOT = Path("data/processed_patient_new")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

def make_patient_id(label, basename):
    """
    label = 0~4
    basename = 9001695L.png → 9001695 / L
    """
    core = basename[:-5]   # 9001695
    side = basename[-5]    # L or R
    return core, side

def process_split(split):
    print(f"▶ Processing split: {split}")

    for label_dir in (RAW_ROOT / split).iterdir():
        if not label_dir.is_dir():
            continue

        label = label_dir.name  # "0", "1", ...

        for img_path in label_dir.iterdir():
            if not img_path.suffix.lower() == ".png":
                continue

            basename = img_path.name  # 9001695L.png
            core, side = make_patient_id(label, basename)

            # Create patient folder
            patient_folder = OUT_ROOT / f"patient_{split}_{label}_{core}"
            patient_folder.mkdir(parents=True, exist_ok=True)

            # Save left/right
            if side == "L":
                out_name = "left.png"
            else:
                out_name = "right.png"

            shutil.copy(img_path, patient_folder / out_name)

    print(f"✔ {split} split done.")


def main():
    for split in ["train", "val", "test"]:
        process_split(split)

    print("\n🎉 All splits processed successfully!")
    print(f"📁 Output saved to: {OUT_ROOT}")


if __name__ == "__main__":
    main()
