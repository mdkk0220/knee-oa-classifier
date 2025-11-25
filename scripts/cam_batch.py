# ============================================================
# scripts/cam_batch.py | v3.0
# ------------------------------------------------------------
# ✅ 목적:
#   - KL 등급별 다중 X-ray 이미지 자동 Grad-CAM 시각화
#   - 학습된 모델(model_best.pth) 기반
# ------------------------------------------------------------
# ⚙️ 실행:
#   python3 scripts/cam_batch.py
# ============================================================

import sys, os, glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np

# 로컬 모듈
from src.models.resnet50 import ResNet50KL
from src.explain.gradcam import GradCAM
from src.explain.viz_utils import overlay_heatmap, save_fig_grid


# ------------------------------------------------------------
# ✅ 설정
# ------------------------------------------------------------
IMG_ROOT = "data/processed/train"
MODEL_PATH = "outputs/resnet50_mj_finetuned/model_best.pth"
OUT_DIR = "outputs/vis/week3/batch"
os.makedirs(OUT_DIR, exist_ok=True)

TR = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

# ------------------------------------------------------------
# ✅ 모델 로드
# ------------------------------------------------------------
model = ResNet50KL(num_classes=5, dropout=0.2)
state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()
target_layer = model.backbone.layer4[-1]
cam = GradCAM(model, target_layer)

# ------------------------------------------------------------
# ✅ KL 등급별 자동 시각화
# ------------------------------------------------------------
for label_dir in sorted(Path(IMG_ROOT).iterdir()):
    if not label_dir.is_dir():
        continue
    label = label_dir.name
    print(f"\n📂 Processing KL{label} ...")

    for img_path in list(label_dir.glob("*.png"))[:5]:  # 각 등급당 5장만
        img = Image.open(img_path).convert("RGB")
        x = TR(img).unsqueeze(0)
        cam_map, pred_cls = cam(x)

        base = np.array(img.resize((224, 224)))
        overlay = overlay_heatmap(base, cam_map.numpy())

        out_path = Path(OUT_DIR) / f"KL{label}_{img_path.stem}.png"
        save_fig_grid([base, overlay], ["Original", "Grad-CAM"], str(out_path))
        print(f"✅ Saved: {out_path}")

print("\n🎯 Batch Grad-CAM visualization complete!")
