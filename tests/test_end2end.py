"""
knee-oa-classifier 6주차: End-to-End 시스템 테스트 (담당: 장미)

기능:
- data/val/** 에서 샘플 이미지 로드
- checkpoints/best.pt 등에서 모델 로드
- 배치 추론 실행 (에러 여부 / 출력 형태 / 속도 확인)
- 결과를 reports/e2e_results.txt 에 저장
"""

import time
from pathlib import Path

import pytest
import torch
from PIL import Image
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VAL_DIR = PROJECT_ROOT / "data" / "val"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = REPORTS_DIR / "e2e_results.txt"


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _find_checkpoint() -> Path:
    if not CHECKPOINT_DIR.exists():
        pytest.skip("checkpoints 디렉토리가 없습니다.")
    cand_names = ["best.pt", "best.pth", "model_best.pt", "model_best.pth"]
    for name in cand_names:
        p = CHECKPOINT_DIR / name
        if p.exists():
            return p
    pytest.skip("사용 가능한 모델 체크포인트 파일이 없습니다. (best.pt 등 필요)")


def _collect_val_images(max_images: int = 16):
    if not VAL_DIR.exists():
        pytest.skip("data/val 디렉토리가 없습니다.")
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    images = []
    labels = []
    for sub in sorted(VAL_DIR.rglob("*")):
        if sub.is_file() and sub.suffix.lower() in exts:
            images.append(sub)
            parent = sub.parent.name.lower()
            num = next((int(c) for c in parent if c.isdigit()), -1)
            labels.append(num)
            if len(images) >= max_images:
                break
    if not images:
        pytest.skip("val 이미지가 없습니다. data/val/KL*/ 에 이미지 몇 장 넣어주세요.")
    return images, labels


def _build_transform(image_size: int = 224):
    import torchvision.transforms as T

    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def _load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    # knee-oa-classifier 기본 구조 기준
    try:
        from src.models.model import build_model  # type: ignore
    except ImportError:
        try:
            from src.model import build_model  # type: ignore
        except ImportError:
            pytest.skip("build_model import 실패. src/models/model.py 구조 확인 필요.")

    model = build_model()
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            k = k[len("model.") :]
        new_state[k] = v

    model.load_state_dict(new_state, strict=False)
    model.to(device)
    model.eval()
    return model


def _write_report(lines):
    text = "[E2E TEST REPORT]\n" + "\n".join(lines) + "\n"
    REPORT_PATH.write_text(text, encoding="utf-8")


@pytest.mark.slow
def test_end_to_end_inference_and_speed():
    device = _get_device()
    ckpt_path = _find_checkpoint()
    image_paths, labels = _collect_val_images(max_images=16)
    transform = _build_transform()

    imgs = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        imgs.append(transform(img))
    batch = torch.stack(imgs, dim=0).to(device)

    model = _load_model(ckpt_path, device)

    start = time.time()
    with torch.no_grad():
        outputs = model(batch)
    end = time.time()

    assert isinstance(outputs, torch.Tensor), "모델 출력이 Tensor 가 아닙니다."
    assert outputs.shape[0] == batch.shape[0], "batch size 불일치"
    assert 2 <= outputs.shape[1] <= 6, f"이상한 클래스 수: {outputs.shape[1]}"
    assert not torch.isnan(outputs).any(), "출력에 NaN 존재"
    assert not torch.isinf(outputs).any(), "출력에 inf 존재"

    elapsed = end - start
    speed_info = f"{len(image_paths)} images in {elapsed:.4f}s (per image {elapsed/len(image_paths):.4f}s)"

    preds = outputs.argmax(dim=1).cpu().numpy().tolist()
    label_arr = np.array(labels)
    valid_mask = label_arr >= 0
    if valid_mask.any():
        acc = (label_arr[valid_mask] == np.array(preds)[valid_mask]).mean()
        acc_line = f"Approx accuracy (rough): {acc:.3f}"
    else:
        acc_line = "유효한 라벨이 없어 accuracy 계산 생략."

    lines = [
        f"Device: {device}",
        f"Checkpoint: {ckpt_path.name}",
        f"Num samples: {len(image_paths)}",
        f"Output shape: {tuple(outputs.shape)}",
        f"Speed: {speed_info}",
        acc_line,
        "Status: PASSED (no exceptions in end-to-end pipeline)",
    ]
    _write_report(lines)
    print("\n".join(lines))
