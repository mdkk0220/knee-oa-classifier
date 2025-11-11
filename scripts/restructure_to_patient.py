# ============================================================
# scripts/restructure_to_patient.py
# ------------------------------------------------------------
# ✅ 목적:
#   data/processed/train/0/... 구조를
#   data/processed_patient/patient_XXXX/pa_ap.png 구조로 변환
#   → 기존 train/val/test 폴더 유지한 채 복사본 생성
# ============================================================

import shutil
from pathlib import Path

SRC_ROOT = Path("data/processed")
DEST_ROOT = Path("data/processed_patient")

splits = ["train", "val", "test"]

for split in splits:
    split_dir = SRC_ROOT / split
    if not split_dir.exists():
        print(f"⚠️ Skip: {split_dir} not found")
        continue

    for label_dir in split_dir.iterdir():
        if not label_dir.is_dir():
            continue

        label = label_dir.name
        img_paths = list(label_dir.glob("*.png"))
        print(f"📂 Processing {split}/{label} ({len(img_paths)} images)")

        for i, img_path in enumerate(img_paths):
            patient_folder = f"patient_{split}_{label}_{i:04d}"
            dest_dir = DEST_ROOT / patient_folder
            dest_dir.mkdir(parents=True, exist_ok=True)

            dest_path = dest_dir / "pa_ap.png"
            shutil.copy(img_path, dest_path)

print("✅ 변환 완료 → data/processed_patient/")
