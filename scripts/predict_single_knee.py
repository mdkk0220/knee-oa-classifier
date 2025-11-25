# ============================================================
# scripts/predict_single_knee.py | 6주차 단일 X-ray 예측 + 모델 전체 정확도 출력
# ------------------------------------------------------------
# - 한 장의 무릎 X-ray로 KL 등급 + 확신도 + Grad-CAM 시각화
# - 모델 전체 성능(Acc, QWK, F1) 함께 출력
# - Week5~6 시스템과 완전 호환
# ============================================================

import sys, os, argparse
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

from src.explain.gradcam import GradCAM
from src.explain.viz_utils import overlay_heatmap, save_fig_grid


# ============================================================
# 🧠 🔥 학습된 모델의 전체 성능(Val/Test) 직접 기입
# ------------------------------------------------------------
# evaluate.py 실행 결과:
# QWK = 0.5869
# Accuracy = 0.5157
# Macro-F1 = 0.3541
# ============================================================
MODEL_ACC = 0.5157
MODEL_QWK = 0.5869
MODEL_F1 = 0.3541


# ============================================================
# 🔥 ResNet50 KL 모델
# ============================================================
class ResNet50KL(nn.Module):
    def __init__(self, pretrained=False, num_classes=5):
        super().__init__()
        self.backbone = models.resnet50(
            weights="IMAGENET1K_V2" if pretrained else None
        )
        self.backbone.fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.fc(x)


# ============================================================
# 🔥 모델 로드 (state_dict 형태 통일)
# ============================================================
def load_model(weight_path):
    device = torch.device("cpu")
    model = ResNet50KL(pretrained=False, num_classes=5)

    ckpt = torch.load(weight_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)

    clean = {}
    for k, v in state_dict.items():
        clean[k.replace("module.", "").replace("backbone.", "")] = v

    model.load_state_dict(clean, strict=False)
    model.to(device).eval()
    return model


# ============================================================
# 🔥 이미지 전처리
# ============================================================
TR = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


def load_image(path):
    img = Image.open(path).convert("RGB")
    x = TR(img).unsqueeze(0)
    return img, x


# ============================================================
# 🔥 메인 (단일 이미지 입력)
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Single Knee KL Prediction")
    parser.add_argument("--img", required=True, help="Path to input X-ray")
    parser.add_argument("--weights", default="outputs/resnet50_lr_full/model_best.pth")
    parser.add_argument("--out", default="outputs/vis/single_test/result.png")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # ---------------------- 입력 이미지 ----------------------
    img_pil, x = load_image(args.img)

    # ---------------------- 모델 로드 ----------------------
    model = load_model(args.weights)

    # Grad-CAM 타겟 레이어
    target_layer = model.backbone.layer4[-1]
    cam = GradCAM(model.backbone, target_layer, use_cuda=False)

    # ---------------------- 예측 ----------------------
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

    # ---------------------- CAM 생성 ----------------------
    cam_map, _ = cam(x)
    cam_map = cam_map.detach().cpu().numpy()

    result_overlay = overlay_heatmap(np.array(img_pil.resize((224, 224))), cam_map)

    # ---------------------- 저장 ----------------------
    save_fig_grid(
        [np.array(img_pil), result_overlay],
        [
            "Original X-ray",
            f"KL {pred.item()} | conf {conf.item():.2f}",
        ],
        args.out,
    )

    # ---------------------- 출력 ----------------------
    print("────────────────────────────")
    print(f"📊 단일 무릎 예측 결과")
    print(f"예측 KL 등급: {pred.item()}")
    print(f"확신도: {conf.item():.2f}")

    print("\n📈 모델 전체 성능 (Val/Test 기준)")
    print(f"Accuracy : {MODEL_ACC * 100:.2f}%")
    print(f"QWK      : {MODEL_QWK:.4f}")
    print(f"Macro-F1 : {MODEL_F1:.4f}")
    print("────────────────────────────")

    print(f"💾 저장 완료: {args.out}")


if __name__ == "__main__":
    main()
