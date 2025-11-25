# ============================================================
# scripts/compare_knees_confidence.py | v3.0 BackboneFC 고정
# ------------------------------------------------------------
# - 좌/우 무릎 이미지 자동 탐색 또는 직접 경로 지정
# - 학습 시 사용한 backbone.fc(Sequential) 구조 그대로 재현
# - KL 0~4 + 확신도 정상 범위로 복구
# - 결과 이미지: outputs/vis/week5/compare_knees_conf.png
# ============================================================

import sys, os, argparse, glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from src.explain.gradcam import GradCAM
from src.explain.viz_utils import overlay_heatmap, save_fig_grid


# ------------------------------ 모델 정의 ------------------------------
class ResNet50KL_BackboneFC(nn.Module):
    """
    학습 시 사용한 구조 그대로 재현:
    - self.backbone: torchvision resnet50
    - self.backbone.fc:
        Linear(2048 -> 512)      # backbone.fc.0
        BatchNorm1d(512)         # backbone.fc.1
        ReLU()                   # backbone.fc.2
        Dropout(p=0.5)           # backbone.fc.3
        Linear(512 -> 5)         # backbone.fc.4
    state_dict 키 예:
    - backbone.fc.0.weight
    - backbone.fc.1.weight / running_mean / running_var ...
    - backbone.fc.4.weight
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = False):
        super().__init__()
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )

        # 학습 당시와 동일한 fc 구조로 덮어쓰기
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 512),     # fc.0
            nn.BatchNorm1d(512),      # fc.1
            nn.ReLU(),                # fc.2
            nn.Dropout(p=0.5),        # fc.3
            nn.Linear(512, num_classes),  # fc.4
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_resnet50_trained(
    weight_path: str = "outputs/resnet50_finetune_combined/model_best.pth",
    num_classes: int = 5,
) -> nn.Module:
    """
    - checkpoint에서 model_state_dict 또는 raw state_dict 읽기
    - module. prefix 있으면 제거
    - ResNet50KL_BackboneFC에 로딩
    """
    device = torch.device("cpu")
    ckpt = torch.load(weight_path, map_location=device)

    state_dict = ckpt.get("model_state_dict", ckpt)

    # module. prefix 정리
    clean_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        clean_state[k] = v

    model = ResNet50KL_BackboneFC(num_classes=num_classes, pretrained=False)

    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    print(f"\n✅ Loaded weights: {weight_path}")
    print(f"ℹ️ Missing: {len(missing)} | Unexpected: {len(unexpected)}")
    if missing:
        print("   - Missing keys:", missing)
    if unexpected:
        print("   - Unexpected keys:", unexpected)

    model.to(device).eval()
    return model


# ------------------------------ 유틸 ------------------------------
TR = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)


def find_pair_or_fallback(left_arg: str | None, right_arg: str | None):
    """
    1) 인자로 경로가 오면 그걸 사용
    2) 아니면 data/processed, data/raw, data 아래에서 *L.png / *R.png 자동 탐색
    3) 그래도 없으면 data/sample_xray.jpg 두 번 사용(최소 검증용)
    """
    if left_arg and right_arg:
        return left_arg, right_arg

    roots = [
        "data/processed",
        "data/raw",
        "data",
    ]
    exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")

    def glob_many(patterns, base):
        files: list[str] = []
        for p in patterns:
            files += glob.glob(os.path.join(base, "**", p), recursive=True)
        return files

    left_paths, right_paths = [], []
    for r in roots:
        if not os.path.isdir(r):
            continue

        left_paths += glob_many([f"*L.{e.split('.')[-1]}" for e in exts], r)
        right_paths += glob_many([f"*R.{e.split('.')[-1]}" for e in exts], r)

    if left_paths and right_paths:
        left_paths.sort()
        right_paths.sort()
        print(f"🔎 Auto-found L: {left_paths[0]}")
        print(f"🔎 Auto-found R: {right_paths[0]}")
        return left_paths[0], right_paths[0]

    sample = "data/sample_xray.jpg"
    if os.path.exists(sample):
        print("⚠️ L/R 쌍을 못 찾음 → sample_xray로 대체")
        return sample, sample

    return None, None


def load_image(path: str):
    img = Image.open(path).convert("RGB")
    return img, TR(img).unsqueeze(0)


# ------------------------------ 메인 ------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare knees with confidence (BackboneFC fixed)"
    )
    parser.add_argument(
        "--weights",
        default="outputs/resnet50_finetune_combined/model_best.pth",
    )
    parser.add_argument("--left", default=None, help="Left knee image path")
    parser.add_argument("--right", default=None, help="Right knee image path")
    parser.add_argument(
        "--out",
        default="outputs/vis/week5/compare_knees_conf.png",
    )
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # 입력 이미지 결정
    left_path, right_path = find_pair_or_fallback(args.left, args.right)
    if not left_path or not right_path:
        print(
            "❌ 테스트 이미지가 없습니다. (--left/--right 지정 또는 data/processed/** 에 L/R 파일 두기)"
        )
        return

    print(f"\n📁 LEFT  = {left_path}")
    print(f"📁 RIGHT = {right_path}")

    left_img, left_x = load_image(left_path)
    right_img, right_x = load_image(right_path)

    # 모델 / GradCAM
    model = build_resnet50_trained(args.weights, num_classes=5)
    backbone = model.backbone
    target_layer = backbone.layer4[-1]

    def predict_with_conf(x):
        cam = GradCAM(backbone, target_layer, use_cuda=False)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)

        cam_map, _ = cam(x)
        if isinstance(cam_map, torch.Tensor):
            cam_map = cam_map.detach().cpu().numpy()

        # KL 0~4 범위 강제
        pred_int = int(pred.item())
        pred_int = max(0, min(4, pred_int))
        conf_float = float(conf.item())
        return pred_int, conf_float, cam_map

    L_pred, L_conf, L_cam = predict_with_conf(left_x)
    R_pred, R_conf, R_cam = predict_with_conf(right_x)

    def mark(c):
        return (
            "신뢰 높음"
            if c >= 0.8
            else ("중간 (의심 예측)" if c >= 0.6 else "낮음 (재검토 필요)")
        )

    L_mark, R_mark = mark(L_conf), mark(R_conf)

    if L_pred > R_pred:
        compare = "왼쪽 무릎이 더 손상된 것으로 예측됩니다."
    elif L_pred < R_pred:
        compare = "오른쪽 무릎이 더 손상된 것으로 예측됩니다."
    else:
        compare = "양쪽 무릎의 손상 정도가 비슷하게 예측됩니다."

    print("\n────────────────────────────")
    print("📊 좌우 무릎 비교 결과 (BackboneFC, CPU)")
    print(f"왼쪽 예측 등급: {L_pred} (확신도: {L_conf:.2f}) → {L_mark}")
    print(f"오른쪽 예측 등급: {R_pred} (확신도: {R_conf:.2f}) → {R_mark}")
    print("────────────────────────────")
    print(f"👉 {compare}")

    # 시각화 저장 (CAM도 같은 backbone 기준)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_fig_grid(
        [
            overlay_heatmap(np.array(left_img.resize((224, 224))), L_cam),
            overlay_heatmap(np.array(right_img.resize((224, 224))), R_cam),
        ],
        [
            f"Left (KL {L_pred}, conf {L_conf:.2f}) {L_mark}",
            f"Right (KL {R_pred}, conf {R_conf:.2f}) {R_mark}",
        ],
        args.out,
    )
    print(f"✅ 시각화 결과 저장: {args.out}")


if __name__ == "__main__":
    main()
