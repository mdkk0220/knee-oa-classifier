# src/explain/viz_utils.py
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import torch

def to_numpy_img(t: torch.Tensor) -> np.ndarray:
    """(C,H,W)[0..1] -> (H,W,C)[0..255] uint8"""
    if t.ndim == 4:
        t = t[0]
    t = t.detach().cpu().clamp(0, 1).numpy()
    t = (t * 255).astype(np.uint8)
    t = np.transpose(t, (1, 2, 0))
    return t


def overlay_heatmap(img_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    RGB 이미지와 [0..1] heatmap을 합성.
    - 기존 JET colormap(파랑~빨강 다색) → 단색 빨강 강조로 변경
    - 모델이 집중한 부분만 자연스럽게 표시 (의료 영상용)
    """
    # Heatmap을 0~255 범위로 변환
    hm = np.uint8(255 * heatmap)
    hm = cv2.resize(hm, (img_rgb.shape[1], img_rgb.shape[0]))

    # 🔴 단색 빨간 채널만 남기기 (Blue/Green=0)
    red_hm = cv2.merge([hm, hm * 0, hm * 0])

    # 알파 블렌딩으로 원본과 합성
    out = cv2.addWeighted(red_hm, alpha, img_rgb, 1 - alpha, 0)
    return out


def save_fig_grid(images: list[np.ndarray], titles: list[str] | None, path: str, cols: int = 2, dpi: int = 140):
    """이미지 여러 장을 그리드로 저장"""
    rows = int(np.ceil(len(images) / cols))
    plt.figure(figsize=(cols * 4, rows * 4), dpi=dpi)
    for i, im in enumerate(images):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(im)
        ax.axis("off")
        if titles:
            ax.set_title(titles[i])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def pil_to_rgb_np(pil: Image.Image) -> np.ndarray:
    return np.array(pil.convert("RGB"))


def read_gray_as_rgb(path: str) -> np.ndarray:
    """흑백 X-ray를 RGB로 확장해서 반환."""
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
