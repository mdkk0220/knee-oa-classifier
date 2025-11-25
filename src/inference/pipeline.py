# ===========================================
# pipeline.py — 정확도(accuracy) 포함 버전
# ===========================================

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import json
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.models.resnet import build_resnet50_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = Path("outputs/resnet50_finetune_combined")
MODEL_PATH = MODEL_DIR / "model_best.pth"

KL_LABELS = [0, 1, 2, 3, 4]

_model: Optional[torch.nn.Module] = None
_model_accuracy: Optional[float] = None


# ------------------------------------------------------------
# 1. 모델 정확도 로딩(학습 로그에서 가져오기)
# ------------------------------------------------------------
def load_model_accuracy() -> Optional[float]:
    global _model_accuracy
    if _model_accuracy is not None:
        return _model_accuracy

    # 1) metrics.json 우선 체크
    metrics_file = MODEL_DIR / "metrics.json"
    if metrics_file.exists():
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "best_val_acc" in data:
                _model_accuracy = float(data["best_val_acc"])
                return _model_accuracy
        except:
            pass

    # 2) train.log 에서 accuracy 파싱
    log_file = MODEL_DIR / "train.log"
    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in reversed(lines):
                if "val_acc" in line.lower():
                    acc = float(line.strip().split()[-1])
                    _model_accuracy = acc
                    return _model_accuracy
        except:
            pass

    # 3) model_best.pth 메타데이터에서 시도
    if MODEL_PATH.exists():
        try:
            ckpt = torch.load(MODEL_PATH, map_location="cpu")
            if "best_val_acc" in ckpt:
                _model_accuracy = float(ckpt["best_val_acc"])
                return _model_accuracy
        except:
            pass

    return None


# ------------------------------------------------------------
# 2. 모델 로딩
# ------------------------------------------------------------
def load_model() -> torch.nn.Module:
    global _model
    if _model is not None:
        return _model

    model = build_resnet50_model(num_classes=len(KL_LABELS))
    state = torch.load(MODEL_PATH, map_location=DEVICE)

    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)

    model.to(DEVICE)
    model.eval()
    _model = model
    return model


# ------------------------------------------------------------
# 3. 전처리
# ------------------------------------------------------------
_infer_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(img: Image.Image) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return _infer_transform(img).unsqueeze(0)


# ------------------------------------------------------------
# 4. 한 장 업로드 → 좌/우 반으로 분해
# ------------------------------------------------------------
def auto_split_left_right_from_single(img: Image.Image):
    w, h = img.size
    left = img.crop((0, 0, w // 2, h))
    right = img.crop((w // 2, 0, w, h))
    return left, right


def infer_left_right_from_files(files: List[Path]):
    if len(files) == 1:
        img = Image.open(files[0])
        L, R = auto_split_left_right_from_single(img)
        return L, R, "left(AutoSplit)", "right(AutoSplit)"

    # 파일명 기반
    left_file = None
    right_file = None
    for f in files:
        n = f.name.lower()
        if any(k in n for k in ["_l", "-l", "left"]):
            left_file = f
        if any(k in n for k in ["_r", "-r", "right"]):
            right_file = f

    # 패턴 없으면 앞 2개
    if left_file is None or right_file is None:
        left_file = files[0]
        right_file = files[1]

    return Image.open(left_file), Image.open(right_file), left_file.name, right_file.name


# ------------------------------------------------------------
# 5. 단일 무릎 예측
# ------------------------------------------------------------
def predict_single(img: Image.Image) -> Dict:
    model = load_model()

    # 🔥 원본 byte 기반 로딩 (Streamlit 왜곡 제거)
    if hasattr(img, "read"):
        img = Image.open(BytesIO(img.read())).convert("RGB")

    # 🔥 원본을 그대로 tf에 넣기 (compare와 동일)
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0].cpu()

    conf, idx = torch.max(prob, dim=0)

    return {
        "kl": int(idx.item()),
        "confidence": float(conf.item()),
        "probs": prob.tolist(),
    }

# ------------------------------------------------------------
# 6. CAM 생성 (뼈대)
# ------------------------------------------------------------
def generate_cam_overlay(img: Image.Image, side: str, save_dir: Path = Path("outputs/ui_cam")) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"cam_{side}.png"

    # TODO: 네 Grad-CAM 코드 연결
    img.save(save_path)  # 임시

    return save_path


# ------------------------------------------------------------
# 7. 전체 파이프라인
# ------------------------------------------------------------
def run_full_pipeline(file_paths: List[str]) -> Dict:
    files = [Path(p) for p in file_paths]

    L_img, R_img, L_name, R_name = infer_left_right_from_files(files)

    L_pred = predict_single_knee(L_img)
    R_pred = predict_single_knee(R_img)

    L_cam = generate_cam_overlay(L_img, "L")
    R_cam = generate_cam_overlay(R_img, "R")

    # 좌/우 비교
    if L_pred["kl"] > R_pred["kl"]:
        summary = f"왼쪽 KL {L_pred['kl']} > 오른쪽 KL {R_pred['kl']} → 왼쪽 진행이 더 심함"
    elif L_pred["kl"] < R_pred["kl"]:
        summary = f"오른쪽 KL {R_pred['kl']} > 왼쪽 KL {L_pred['kl']} → 오른쪽 진행이 더 심함"
    else:
        summary = f"양측 KL {L_pred['kl']} → 진행 정도 유사"

    # 🔥 정확도 포함
    accuracy = load_model_accuracy()
    if accuracy is None:
        accuracy = -1  # fallback

    return {
        "accuracy": accuracy,   # 🔥 UI가 쓸 전체 모델 정확도
        "left": {
            "filename": L_name,
            "kl": L_pred["kl"],
            "confidence": L_pred["confidence"],
            "cam_path": str(L_cam),
        },
        "right": {
            "filename": R_name,
            "kl": R_pred["kl"],
            "confidence": R_pred["confidence"],
            "cam_path": str(R_cam),
        },
        "summary": summary,
    }
