# ============================================================
# scripts/make_manifest_patient.py
# ------------------------------------------------------------
# ✅ 목적:
#   data/processed_patient/ 폴더를 스캔해서
#   filepath, label, patient_id, split 컬럼 포함 manifest 생성
# ============================================================

from pathlib import Path
import pandas as pd

ROOT = Path("data/processed_patient")
out_csv = Path("metadata/dataset_manifest.csv")

records = []

for p in ROOT.iterdir():
    if not p.is_dir():
        continue

    # 예시: patient_train_0_0003
    parts = p.name.split("_")
    if len(parts) < 4:
        print(f"⚠️ Skip: {p.name} (unexpected folder format)")
        continue

    split, label, pid = parts[1], parts[2], parts[3]
    img_path = p / "pa_ap.png"

    if not img_path.exists():
        print(f"⚠️ Skip missing: {img_path}")
        continue

    records.append({
        "filepath": str(img_path),
        "label": int(label),
        "patient_id": pid,
        "split": split,
    })

df = pd.DataFrame(records)
out_csv.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_csv, index=False)

print(f"✅ Manifest saved → {out_csv}")
print(f"📊 Total samples: {len(df)}")
print(df.head())
