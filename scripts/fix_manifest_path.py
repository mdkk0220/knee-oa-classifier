# ============================================================
# scripts/fix_manifest_path.py
# ------------------------------------------------------------
# ✅ manifest 내 중복 경로 제거:
#    "data/processed_patient/data/processed_patient" → "data/processed_patient"
#    "data/processed_patient/" → ""
# ============================================================

import pandas as pd

csv_path = "metadata/dataset_manifest.csv"
df = pd.read_csv(csv_path)

# 경로 중복 부분 제거
df["filepath"] = (
    df["filepath"]
    .str.replace("data/processed_patient/data/processed_patient", "data/processed_patient", regex=False)
    .str.replace("data/processed_patient/", "", regex=False)
)

out_path = "metadata/dataset_manifest_fixed.csv"
df.to_csv(out_path, index=False)

print(f"✅ Fixed manifest saved → {out_path}")
print(f"📊 총 {len(df)}개 경로 수정 완료")
print(df.head(3))
