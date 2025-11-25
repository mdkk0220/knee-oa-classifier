# scripts/make_manifest_from_processed.py

import csv
from pathlib import Path

ROOT = Path("data/processed_patient_new")
OUT_CSV = Path("metadata/dataset_manifest_patient.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

def main():
    rows = []

    for patient_dir in sorted(ROOT.iterdir()):
        if not patient_dir.is_dir():
            continue

        name = patient_dir.name  # patient_train_0_9001695
        parts = name.split("_")

        split = parts[1]  # train / val / test
        label = parts[2]  # 0~4

        left_path = patient_dir / "left.png"
        right_path = patient_dir / "right.png"

        rows.append({
            "patient_id": name,
            "left": str(left_path),
            "right": str(right_path),
            "label": int(label),
            "split": split,
        })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "left", "right", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)

    print("🎉 Manifest created:", OUT_CSV)


if __name__ == "__main__":
    main()
