# ============================================================
# 📁 scripts/compare_knees_auto_pairs.py
# ------------------------------------------------------------
# 6주차 명규 — test 전체 L/R 자동 비교 + CSV 저장
# ------------------------------------------------------------
# 기능
#  - data/raw/archive2/test/ 아래의 KL 0~4 폴더에서
#    *L.png / *R.png 패턴으로 L/R 페어 자동 매칭
#  - 새로 학습한 모델(outputs/resnet50_lr_full/model_best.pth)로
#    각 쌍에 대해 KL 예측 + 확신도 + Grad-CAM 생성
#  - 비교 결과를 이미지 + CSV로 저장
#    CSV: outputs/vis/auto_pairs/results_auto_pairs.csv
# ============================================================

import os
import sys
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

# 프로젝트 루트 & src 경로 추가
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.explain.gradcam import GradCAM
from src.explain.viz_utils import overlay_heatmap, save_fig_grid
from src.data.transforms import get_val_aug
from src.models.resnet50 import ResNet50KL


# ============================================================
# 🔧 기본 설정
# ============================================================
DEFAULT_IMG_ROOT = Path("data/raw/archive2/test")   # test/0 ~ test/4
DEFAULT_MODEL_PATH = "outputs/resnet50_lr_full/model_best.pth"
DEFAULT_OUT_DIR = Path("outputs/vis/auto_pairs")
DEFAULT_CSV_PATH = DEFAULT_OUT_DIR / "results_auto_pairs.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 🔧 모델 로드
# ============================================================
def load_model(weight_path: str) -> ResNet50KL:
    print(f"🔄 Loading model from: {weight_path}")
    model = ResNet50KL(num_classes=5)
    state = torch.load(weight_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


# ============================================================
# 🔧 L/R 페어 탐색
# ============================================================
def find_lr_pairs(img_root: Path):
    """
    img_root: data/raw/archive2/test
      - 그 아래에 0,1,2,3,4 폴더 존재
      - 각 폴더 안에 9003175L.png / 9003175R.png 형태

    반환: 리스트[
        {
          "patient_id": "9003175",
          "label": 0,
          "left_path": Path(...L.png),
          "right_path": Path(...R.png),
        },
        ...
    ]
    """
    pairs = []

    for grade in range(5):
        grade_dir = img_root / str(grade)
        if not grade_dir.is_dir():
            continue

        # id -> {"L": path, "R": path}
        table = {}

        for img_path in grade_dir.glob("*.png"):
            name = img_path.stem  # 예: "9888797L"
            if len(name) < 2:
                continue

            side = name[-1]  # 'L' 또는 'R'
            pid = name[:-1]  # "9888797"

            if side not in ("L", "R"):
                continue

            if pid not in table:
                table[pid] = {"L": None, "R": None}

            table[pid][side] = img_path

        # L/R 둘 다 있는 것만 페어로 추가
        for pid, lr in table.items():
            if lr["L"] is None or lr["R"] is None:
                continue
            pairs.append(
                {
                    "patient_id": pid,
                    "label": grade,
                    "left_path": lr["L"],
                    "right_path": lr["R"],
                }
            )

    return pairs


# ============================================================
# 🔧 예측 + CAM
# ============================================================
def predict_and_cam(model, img_rgb: np.ndarray, transform):
    """
    img_rgb: (H,W,3) RGB numpy
    transform: get_val_aug(512) 같은 Albumentations Compose
    """
    tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)

    # 예측
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

    # GradCAM (backbone 기준 layer4의 마지막 블록)
    target_layer = model.backbone.layer4[-1]
    cam_extractor = GradCAM(model.backbone, target_layer, use_cuda=(DEVICE.type == "cuda"))

    try:
        cam_map, _ = cam_extractor(tensor)
        if isinstance(cam_map, torch.Tensor):
            cam_map = cam_map.squeeze().detach().cpu().numpy()
        # 0~1 정규화
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


def confidence_label(score: float) -> str:
    if score >= 0.80:
        return "신뢰 높음"
    elif score >= 0.60:
        return "중간"
    return "낮음 (재검토)"


# ============================================================
# 🔥 메인: 전체 test 세트 자동 처리
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Auto L/R comparison for archive2 test set")
    parser.add_argument("--root", default=str(DEFAULT_IMG_ROOT), help="test root (e.g., data/raw/archive2/test)")
    parser.add_argument("--weights", default=DEFAULT_MODEL_PATH, help="model_best.pth path")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR), help="output directory for images")
    parser.add_argument("--csv", default=str(DEFAULT_CSV_PATH), help="CSV result path")
    args = parser.parse_args()

    img_root = Path(args.root)
    out_dir = Path(args.out_dir)
    csv_path = Path(args.csv)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) L/R 페어 스캔
    pairs = find_lr_pairs(img_root)
    print(f"🔍 Found {len(pairs)} L/R pairs in test set")

    if len(pairs) == 0:
        print("❌ L/R 페어를 하나도 찾지 못했습니다. 파일 이름/경로를 다시 확인하세요.")
        return

    # 2) 모델 & 변환 준비
    model = load_model(args.weights)
    transform = get_val_aug(512)

    rows = []

    # 3) 전체 페어 순회
    for idx, info in enumerate(pairs, start=1):
        pid = info["patient_id"]
        label = info["label"]
        left_path = info["left_path"]
        right_path = info["right_path"]

        print(f"[{idx}/{len(pairs)}] ▶ {pid} (label={label})")

        # 이미지 로드 (흑백 → RGB)
        left_gray = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
        right_gray = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)

        if left_gray is None or right_gray is None:
            print(f"❌ 이미지 로드 실패: {left_path} / {right_path}")
            continue

        left_rgb = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2RGB)
        right_rgb = cv2.cvtColor(right_gray, cv2.COLOR_GRAY2RGB)

        # 예측 + CAM
        pL, cL, camL = predict_and_cam(model, left_rgb, transform)
        pR, cR, camR = predict_and_cam(model, right_rgb, transform)

        # 비교 문구
        if pL > pR:
            worse_side = "left"
            compare_text = "왼쪽 무릎이 더 손상"
        elif pL < pR:
            worse_side = "right"
            compare_text = "오른쪽 무릎이 더 손상"
        else:
            worse_side = "equal"
            compare_text = "양쪽 비슷한 손상도"

        # 시각화 이미지 저장
        left_overlay = overlay_heatmap(left_rgb, camL)
        right_overlay = overlay_heatmap(right_rgb, camR)

        img_out_path = out_dir / f"{label}_{pid}_pair.png"
        save_fig_grid(
            images=[left_rgb, left_overlay, right_rgb, right_overlay],
            titles=[
                f"LEFT  KL-{pL} ({cL:.2f}) {confidence_label(cL)}",
                "LEFT CAM",
                f"RIGHT KL-{pR} ({cR:.2f}) {confidence_label(cR)}",
                "RIGHT CAM",
            ],
            path=str(img_out_path),
            cols=2,
            dpi=140,
        )

        # CSV 한 줄 추가
        rows.append(
            [
                pid,
                label,
                pL,
                f"{cL:.4f}",
                pR,
                f"{cR:.4f}",
                worse_side,
                compare_text,
                str(img_out_path),
            ]
        )

    # 4) CSV 저장
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "patient_id",
                "true_label",
                "left_pred",
                "left_conf",
                "right_pred",
                "right_conf",
                "worse_side",
                "compare_result",
                "image_path",
            ]
        )
        writer.writerows(rows)

    print("────────────────────────────")
    print(f"📄 CSV 저장 완료: {csv_path}")
    print(f"🖼 시각화 폴더   : {out_dir}")
    print("✅ test 세트 전체 좌/우 비교 자동 처리 완료")


if __name__ == "__main__":
    main()
