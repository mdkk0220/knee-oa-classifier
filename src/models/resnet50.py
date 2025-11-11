# scripts/compare_knees_confidence.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pathlib import Path

from src.explain.gradcam import GradCAM
from src.explain.viz_utils import overlay_heatmap, save_fig_grid


# -------------------------------------------------
# ✅ 1. ResNet50KL 구조 직접 정의 (학습 당시와 동일)
# -------------------------------------------------
class ResNet50KL(nn.Module):
    def __init__(self, pretrained=False, num_classes=5):
        super().__init__()
        self.backbone = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        self.backbone.fc = nn.Identity()  # 마지막 FC 제거
        self.fc = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.fc(x)


# -------------------------------------------------
# ✅ 2. 학습된 모델 불러오기
# -------------------------------------------------
def build_resnet50_trained(weight_path="outputs/resnet50_finetune_combined/model_best.pth"):
    model = ResNet50KL(pretrained=False, num_classes=5)
    state_dict = torch.load(weight_path, map_location="cpu")

    # prefix 정리
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "").replace("backbone.", "")
        new_state_dict[new_key] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"✅ Loading trained model from {weight_path}")
    print(f"ℹ️ Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    model.eval()
    return model


# -------------------------------------------------
# 3. 이미지 전처리
# -------------------------------------------------
TR = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])


# -------------------------------------------------
# 4. 좌우 무릎 비교 + 확신도 분석
# -------------------------------------------------
def main():
    left_path = "data/processed/train/2/9002411L.png"
    right_path = "data/processed/train/2/9002411R.png"
    out_path = "outputs/vis/week5/compare_knees_conf.png"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # 이미지 로드
    try:
        left_img = Image.open(left_path).convert("RGB")
        right_img = Image.open(right_path).convert("RGB")
    except FileNotFoundError:
        print("❌ 왼쪽 또는 오른쪽 이미지 파일을 찾을 수 없습니다.")
        return

    left_x = TR(left_img).unsqueeze(0)
    right_x = TR(right_img).unsqueeze(0)

    model = build_resnet50_trained("outputs/resnet50_finetune_combined/model_best.pth")
    target_layer = model.backbone.layer4[-1]
    cam = GradCAM(model.backbone, target_layer)

    def predict_with_conf(x):
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
        cam_map, _ = cam(x)
        return pred.item(), conf.item(), cam_map

    left_pred, left_conf, left_cam = predict_with_conf(left_x)
    right_pred, right_conf, right_cam = predict_with_conf(right_x)

    # 확신도 등급
    def confidence_mark(conf):
        if conf >= 0.8:
            return "신뢰 높음"
        elif conf >= 0.6:
            return "중간 (의심 예측)"
        else:
            return "낮음 (재검토 필요)"

    left_mark = confidence_mark(left_conf)
    right_mark = confidence_mark(right_conf)

    # 비교 결과
    if left_pred > right_pred:
        compare_text = "왼쪽 무릎이 더 손상된 것으로 예측됩니다."
    elif left_pred < right_pred:
        compare_text = "오른쪽 무릎이 더 손상된 것으로 예측됩니다."
    else:
        compare_text = "양쪽 무릎의 손상 정도가 비슷하게 예측됩니다."

    # 콘솔 출력
    print("────────────────────────────")
    print("📊 좌우 무릎 비교 결과")
    print(f"왼쪽 예측 등급: {left_pred} (확신도: {left_conf:.2f}) → {left_mark}")
    print(f"오른쪽 예측 등급: {right_pred} (확신도: {right_conf:.2f}) → {right_mark}")
    print("────────────────────────────")
    print(f"👉 {compare_text}")
    print(f"✅ 시각화 결과 저장: {out_path}")

    # 시각화 저장
    save_fig_grid(
        [
            overlay_heatmap(np.array(left_img.resize((224, 224))), left_cam.numpy()),
            overlay_heatmap(np.array(right_img.resize((224, 224))), right_cam.numpy()),
        ],
        [
            f"Left Knee (KL {left_pred}, conf {left_conf:.2f}) {left_mark}",
            f"Right Knee (KL {right_pred}, conf {right_conf:.2f}) {right_mark}",
        ],
        out_path,
    )


if __name__ == "__main__":
    main()
