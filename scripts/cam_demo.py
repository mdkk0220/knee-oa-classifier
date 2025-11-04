# ============================================================
# scripts/cam_demo.py | v3.0 (3주차 확장 버전)
# ------------------------------------------------------------
# ✅ 목적:
#   - 학습된 ResNet50KL 모델(model_best.pth) 기반 Grad-CAM 시각화
#   - 단일 이미지 테스트용
# ------------------------------------------------------------
# ⚙️ 실행:
#   python3 scripts/cam_demo.py
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np

# 로컬 모듈
from src.explain.gradcam import GradCAM
from src.explain.viz_utils import overlay_heatmap, save_fig_grid
from src.models.resnet50 import ResNet50KL


# ------------------------------------------------------------
# ✅ 학습된 모델 로드 함수
# ------------------------------------------------------------
def build_resnet50_trained(weight_path: str):
    model = ResNet50KL(num_classes=5, dropout=0.2)
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✅ Loaded trained model from: {weight_path}")
    return model


# ------------------------------------------------------------
# ✅ 변환 정의
# ------------------------------------------------------------
TR = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])


# ------------------------------------------------------------
# ✅ 실행 함수
# ------------------------------------------------------------
def main():
    # 🔹 입력 이미지 및 출력 경로 지정
    img_path = "data/processed/train/2/9000296R.png"   # 예시
    MODEL_PATH = "outputs/resnet50_mj_finetuned/model_best.pth"
    out_path = "outputs/vis/week3/cam_sample.png"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # 🔹 이미지 불러오기
    try:
        img = Image.open(img_path).convert("RGB")
    except FileNotFoundError:
        print(f"❌ 이미지 파일을 찾을 수 없습니다: {img_path}")
        return

    x = TR(img).unsqueeze(0)

    # 🔹 모델 로드
    model = build_resnet50_trained(model_path)
    target_layer = model.backbone.layer4[-1]  # Grad-CAM 타깃 레이어 지정

    # 🔹 Grad-CAM 실행
    cam = GradCAM(model, target_layer)
    cam_map, pred_cls = cam(x)

    # 🔹 시각화 및 저장
    base = np.array(img.resize((224, 224)))
    overlay = overlay_heatmap(base, cam_map.numpy())
    save_fig_grid([base, overlay],
                  ["Original", "Grad-CAM"],
                  out_path)

    print(f"✅ Saved: {out_path}")
    print(f"✅ Predicted Class Index: {pred_cls}")


if __name__ == "__main__":
    main()
