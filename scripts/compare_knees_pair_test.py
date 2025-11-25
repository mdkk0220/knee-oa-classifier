# ============================================================
# 📁 scripts/compare_knees_pair_test.py
# ------------------------------------------------------------
# 좌/우 한 쌍 X-ray 비교 테스트 (Grad-CAM 포함)
# - 모드 1: --left / --right 로 L/R 이미지를 직접 지정
# - 모드 2: --img 한 장 주면 내부에서 자동으로 L/R 분리 후 비교
# ------------------------------------------------------------
# ✅ 모델: outputs/resnet50_lr_full/model_best.pth (ResNet50KL)
# ✅ Grad-CAM: model.backbone.layer4 기준
# ============================================================

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import argparse

# ------------------------------------------------------------
# 🔧 src 경로 세팅
# ------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from src.explain.gradcam import GradCAM
from src.explain.viz_utils import overlay_heatmap, save_fig_grid
from src.data.transforms import get_val_aug
from src.models.resnet50 import ResNet50KL

MODEL_PATH = "outputs/resnet50_lr_full/model_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 🔧 모델 로드
# ============================================================
def load_model():
    print(f"🔄 Loading model from: {MODEL_PATH}")
    model = ResNet50KL(num_classes=5)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


# ============================================================
# 🔧 단일 X-ray(한 장) → 좌/우 이미지로 분리
# ============================================================
def split_single_lr(img_path: Path):
    """
    하나의 무릎 X-ray(정면 양쪽 다리 포함)를
    화면 기준 좌/우 절반으로 잘라서 L/R 이미지를 만든다.
    - 입력: 원본 X-ray 경로
    - 출력: left_rgb, right_rgb (둘 다 HxWx3 RGB numpy 배열)
    """
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"이미지 로딩 실패: {img_path}")

    h, w = gray.shape
    mid = w // 2

    # 화면 기준 왼쪽/오른쪽 절반
    left_gray = gray[:, :mid]
    right_gray = gray[:, mid:]

    left_rgb = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2RGB)
    right_rgb = cv2.cvtColor(right_gray, cv2.COLOR_GRAY2RGB)

    return left_rgb, right_rgb


# ============================================================
# 🔧 예측 + Confidence + CAM 처리
# ============================================================
def predict_and_cam(model, img_rgb, transform):
    # albumentations 변환 적용
    t = transform(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)

    # 예측
    with torch.no_grad():
        logits = model(t)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

    # Grad-CAM: ResNet50KL의 backbone.layer4를 타겟으로 사용
    cam_extractor = GradCAM(model.backbone, target_layer=model.backbone.layer4, use_cuda=False)

    try:
        cam_map, _ = cam_extractor(t)
        if isinstance(cam_map, torch.Tensor):
            cam_map = cam_map.squeeze().detach().cpu().numpy()
        cam_min, cam_max = cam_map.min(), cam_map.max()
        if cam_max > cam_min:
            cam_norm = (cam_map - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_map, dtype=np.float32)
    except Exception as e:
        print(f"⚠️ Grad-CAM 오류: {e}")
        h, w = img_rgb.shape[:2]
        cam_norm = np.zeros((h, w), dtype=np.float32)

    return int(pred.item()), float(conf.item()), cam_norm


# ============================================================
# 🔧 Confidence 라벨
# ============================================================
def confidence_label(score):
    if score >= 0.80:
        return "✅ 신뢰 높음"
    elif score >= 0.60:
        return "⚠️ 중간"
    return "❌ 낮음 (재검토)"


# ============================================================
# 🔥 메인
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="L/R 쌍 비교 테스트")
    parser.add_argument("--left", help="이미 분리된 LEFT 이미지 경로", default=None)
    parser.add_argument("--right", help="이미 분리된 RIGHT 이미지 경로", default=None)
    parser.add_argument("--img", help="한 장짜리 X-ray (자동 L/R 분리용)", default=None)
    args = parser.parse_args()

    use_single = args.img is not None

    # -------------------------------
    # 1) 입력 검증
    # -------------------------------
    if use_single:
        img_path = Path(args.img)
        if not img_path.exists():
            raise FileNotFoundError(f"--img 파일을 찾을 수 없습니다: {img_path}")
        print("✨ L/R 페어 비교 시작 (단일 X-ray 자동 분리 모드)")
        print(f"  IMG : {img_path}")
    else:
        if args.left is None or args.right is None:
            raise ValueError("--img 또는 (--left, --right) 중 하나는 반드시 지정해야 합니다.")
        left_path = Path(args.left)
        right_path = Path(args.right)
        if not left_path.exists():
            raise FileNotFoundError(f"LEFT 이미지 없음: {left_path}")
        if not right_path.exists():
            raise FileNotFoundError(f"RIGHT 이미지 없음: {right_path}")
        print("✨ L/R 페어 비교 시작 (직접 지정 모드)")
        print(f"  LEFT : {left_path}")
        print(f"  RIGHT: {right_path}")

    # -------------------------------
    # 2) 모델 & 변환 준비
    # -------------------------------
    model = load_model()
    transform = get_val_aug(512)

    # -------------------------------
    # 3) L/R 이미지 준비
    # -------------------------------
    if use_single:
        # 단일 X-ray → 자동 L/R 분리
        left_rgb, right_rgb = split_single_lr(img_path)
        left_stem = img_path.stem + "_L"
        right_stem = img_path.stem + "_R"
    else:
        # 기존처럼 L/R 파일에서 로드
        left_gray = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
        right_gray = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)

        if left_gray is None:
            raise ValueError(f"LEFT 이미지 로딩 실패: {left_path}")
        if right_gray is None:
            raise ValueError(f"RIGHT 이미지 로딩 실패: {right_path}")

        left_rgb = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2RGB)
        right_rgb = cv2.cvtColor(right_gray, cv2.COLOR_GRAY2RGB)

        left_stem = left_path.stem
        right_stem = right_path.stem

    # -------------------------------
    # 4) 예측 + CAM
    # -------------------------------
    pL, cL, camL = predict_and_cam(model, left_rgb, transform)
    pR, cR, camR = predict_and_cam(model, right_rgb, transform)

    left_overlay = overlay_heatmap(left_rgb, camL)
    right_overlay = overlay_heatmap(right_rgb, camR)

    title_left = f"LEFT  KL-{pL} ({cL:.2f}) {confidence_label(cL)}"
    title_right = f"RIGHT KL-{pR} ({cR:.2f}) {confidence_label(cR)}"

    # -------------------------------
    # 5) 저장 경로 & 출력
    # -------------------------------
    save_dir = Path("outputs/vis/pair_test")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{left_stem}_vs_{right_stem}.png"

    save_fig_grid(
        images=[left_rgb, left_overlay, right_rgb, right_overlay],
        titles=[title_left, "LEFT CAM", title_right, "RIGHT CAM"],
        path=str(save_path),
        cols=2,
        dpi=140,
    )

    print("────────────────────────────")
    print(f"LEFT  → KL {pL}, conf={cL:.2f}  {confidence_label(cL)}")
    print(f"RIGHT → KL {pR}, conf={cR:.2f}  {confidence_label(cR)}")
    print("────────────────────────────")
    if pL > pR:
        print("👉 왼쪽 무릎이 더 손상")
    elif pL < pR:
        print("👉 오른쪽 무릎이 더 손상")
    else:
        print("👉 양쪽 비슷한 손상도")
    print(f"💾 저장 완료: {save_path}")


if __name__ == "__main__":
    main()
