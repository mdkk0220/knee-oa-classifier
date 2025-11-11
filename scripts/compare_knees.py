# scripts/compare_knees.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from src.explain.gradcam import GradCAM
from src.explain.viz_utils import overlay_heatmap, save_fig_grid
from pathlib import Path

# -----------------------------
# 1. 모델 빌드 (ImageNet pretrained)
# -----------------------------
def build_resnet50_imagenet():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    return model

# -----------------------------
# 2. 이미지 전처리
# -----------------------------
TR = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

# -----------------------------
# 3. 메인 로직
# -----------------------------
def main():
    left_path = "data/processed/train/2/9000296L.png"   # 왼쪽 예시
    right_path = "data/processed/train/2/9000296R.png"  # 오른쪽 예시
    out_path = "outputs/vis/week4/compare_knees.png"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # 이미지 로드
    try:
        left_img = Image.open(left_path).convert("RGB")
        right_img = Image.open(right_path).convert("RGB")
    except FileNotFoundError:
        print("❌ 왼쪽 또는 오른쪽 이미지 파일을 찾을 수 없습니다.")
        return

    # 전처리
    left_x = TR(left_img).unsqueeze(0)
    right_x = TR(right_img).unsqueeze(0)

    # 모델 및 Grad-CAM 설정
    model = build_resnet50_imagenet()
    target_layer = model.layer4[-1]
    cam = GradCAM(model, target_layer)

    # 예측 + CAM 생성
    left_cam, left_cls = cam(left_x)
    right_cam, right_cls = cam(right_x)

    # 시각화
    left_base = np.array(left_img.resize((224, 224)))
    right_base = np.array(right_img.resize((224, 224)))

    left_overlay = overlay_heatmap(left_base, left_cam.numpy())
    right_overlay = overlay_heatmap(right_base, right_cam.numpy())

    # -----------------------------
    # 4. 좌/우 비교 로직
    # -----------------------------
    if left_cls > right_cls:
        compare_text = "왼쪽 무릎이 더 손상된 것으로 예측됩니다."
    elif left_cls < right_cls:
        compare_text = "오른쪽 무릎이 더 손상된 것으로 예측됩니다."
    else:
        compare_text = "양쪽 무릎의 손상 정도가 비슷하게 예측됩니다."

    print("✅", compare_text)
    print(f"왼쪽 예측 등급: {left_cls}, 오른쪽 예측 등급: {right_cls}")

    # -----------------------------
    # 5. 결과 저장
    # -----------------------------
    save_fig_grid(
        [left_overlay, right_overlay],
        ["Left Knee", "Right Knee"],
        out_path
    )
    print(f"✅ 시각화 결과 저장 완료: {out_path}")

if __name__ == "__main__":
    main()
