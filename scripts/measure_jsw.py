# ============================================================
# measure_jsw.py | Medial JSW 측정 (안정 버전, KL 분리)
# ------------------------------------------------------------
# - 입력: 좌/우 무릎 AP X-ray (각각 한 장씩)
# - 내측(medial) ROI에서 여러 세로 컬럼을 훑어서
#   femur 하연 ~ tibia 상연 간격(px)을 측정
# - 가장 좁은(하위 20% 근처) 간격을 대표 JSW로 사용
# - 결과: 각 무릎에 짧은 가로선 + 점선 세로선 + JSW(mm) 텍스트
# - KL 등급은 이 스크립트에서 사용하지 않고,
#   기존 compare_knees_confidence.py 쪽에서 그대로 사용
# ============================================================

import cv2
import numpy as np
from pathlib import Path


# ------------------------------------------------------------
# 1. 내측(medial) ROI 설정
# ------------------------------------------------------------
def get_medial_roi(h: int, w: int, side: str):
    """
    h, w: 단일 무릎 이미지 높이/너비
    side: 'L' (왼쪽 무릎), 'R' (오른쪽 무릎)
    반환: (x1, x2, y1, y2)  # 내측 관절 간격이 있을 법한 사각형
    """
    y1 = int(h * 0.30)
    y2 = int(h * 0.80)

    side = side.upper()
    if side == "L":
        # 왼쪽 무릎 → 이미지 오른쪽이 내측
        x1 = int(w * 0.45)
        x2 = int(w * 0.95)
    else:
        # 오른쪽 무릎 → 이미지 왼쪽이 내측
        x1 = int(w * 0.05)
        x2 = int(w * 0.55)

    return x1, x2, y1, y2


# ------------------------------------------------------------
# 2. 다중 컬럼 기반 JSW(px) 계산
# ------------------------------------------------------------
def compute_jsw_px(gray: np.ndarray, side: str):
    """
    gray: 단일 무릎 grayscale 이미지
    side: 'L' / 'R'
    반환: (jsw_px, x_mid, y_femur, y_tibia)
    """
    h, w = gray.shape

    # 전처리: 블러 + CLAHE + Canny
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)
    edges = cv2.Canny(enhanced, 40, 120)

    # 내측 ROI
    x1, x2, y1, y2 = get_medial_roi(h, w, side)
    roi = edges[y1:y2, x1:x2]
    roi_h, roi_w = roi.shape

    candidates = []

    # 허용 두께 범위(px)
    min_px = max(3, int(roi_h * 0.03))
    max_px = int(roi_h * 0.30)

    # ROI 가로 15%~85% 구간에서 여러 컬럼 스캔
    start_c = int(roi_w * 0.15)
    end_c = int(roi_w * 0.85)
    step = max(1, roi_w // 30)  # 대략 20~30개 컬럼

    for cx in range(start_c, end_c, step):
        col = roi[:, cx]
        ys = np.where(col > 0)[0]
        if len(ys) < 2:
            continue

        top_local = int(ys[0])
        bottom_local = int(ys[-1])
        dist = bottom_local - top_local

        if dist < min_px or dist > max_px:
            continue

        top = y1 + top_local
        bottom = y1 + bottom_local
        gx = x1 + cx
        mid = (top + bottom) / 2.0

        # 세로 위치가 너무 위/아래가 아닌지 체크
        if not (h * 0.30 <= mid <= h * 0.80):
            continue

        candidates.append((float(dist), gx, top, bottom))

    if not candidates:
        raise RuntimeError("관절선을 안정적으로 찾지 못했습니다.")

    # 관절 간격 분포에서 하위 20% 근처의 좁은 간격을 대표값으로 선택
    dists = np.array([c[0] for c in candidates], dtype=np.float32)
    target = np.percentile(dists, 20)

    near = [c for c in candidates if c[0] <= target * 1.3]
    if not near:
        best = min(candidates, key=lambda c: c[0])
    else:
        best = min(near, key=lambda c: c[0])

    jsw_px, x_mid, y_femur, y_tibia = best
    return jsw_px, int(x_mid), int(y_femur), int(y_tibia)


# ------------------------------------------------------------
# 3. 병원 스타일 오버레이 (선 + 점선 + 텍스트)
# ------------------------------------------------------------
def draw_jsw_overlay(
    img_bgr: np.ndarray,
    x_mid: int,
    y_femur: int,
    y_tibia: int,
    jsw_mm: float,
    side: str,
) -> np.ndarray:
    overlay = img_bgr.copy()
    h, w, _ = overlay.shape

    line_color = (255, 220, 120)   # 밝은 청록톤
    text_color = (255, 255, 255)   # 흰색 글씨

    # 가로선 길이
    line_len = int(w * 0.13)
    x1 = max(0, x_mid - line_len // 2)
    x2 = min(w - 1, x_mid + line_len // 2)

    # Femur / Tibia 가로선
    cv2.line(overlay, (x1, y_femur), (x2, y_femur), line_color, 3, cv2.LINE_AA)
    cv2.line(overlay, (x1, y_tibia), (x2, y_tibia), line_color, 3, cv2.LINE_AA)

    # 수직 점선
    dash, gap = 7, 6
    y = y_femur
    while y < y_tibia:
        y2 = min(y + dash, y_tibia)
        cv2.line(overlay, (x_mid, y), (x_mid, y2), line_color, 2, cv2.LINE_AA)
        y += dash + gap

    # 텍스트 (각 무릎 내부에서만, 잘리지 않게)
    label = f"{jsw_mm:.2f} mm"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.85
    thickness = 2

    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
    y_text = max(text_h + 5, y_femur - 18)

    side = side.upper()
    if side == "L":
        x_base = int(w * 0.18)
    else:
        x_base = int(w * 0.58)

    # 이미지 안에 확실히 들어오도록 클램프
    x_text = max(8, min(x_base, w - text_w - 8))

    cv2.putText(
        overlay,
        label,
        (x_text, y_text),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )

    return overlay


# ------------------------------------------------------------
# 4. 단일 무릎 처리
# ------------------------------------------------------------
def process_single(path: str, side: str, px_to_mm: float = 0.1):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"이미지 로드 실패: {path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    jsw_px, x_mid, y_femur, y_tibia = compute_jsw_px(gray, side)
    jsw_mm = jsw_px * px_to_mm

    overlay = draw_jsw_overlay(img, x_mid, y_femur, y_tibia, jsw_mm, side)
    return jsw_mm, overlay


# ------------------------------------------------------------
# 5. 좌/우 JSW 비교 후 한 장으로 저장
# ------------------------------------------------------------
def compare_jsw(left_path: str, right_path: str, save_path: str, px_to_mm: float = 0.1):
    left_jsw, left_img = process_single(left_path, "L", px_to_mm)
    right_jsw, right_img = process_single(right_path, "R", px_to_mm)

    # 세로 크기 맞춰서 좌우 이어붙이기
    h = min(left_img.shape[0], right_img.shape[0])
    left_resized = cv2.resize(left_img, (int(left_img.shape[1] * h / left_img.shape[0]), h))
    right_resized = cv2.resize(right_img, (int(right_img.shape[1] * h / right_img.shape[0]), h))

    combined = np.hstack([left_resized, right_resized])

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, combined)

    print("────────────────────────────")
    print(f"Left  Medial JSW = {left_jsw:.2f} mm")
    print(f"Right Medial JSW = {right_jsw:.2f} mm")
    print("저장:", save_path)


# ------------------------------------------------------------
# 6. 실행부 (네가 말한 전처리 데이터 경로 기준)
# ------------------------------------------------------------
if __name__ == "__main__":
    left_img = "data/raw/archive2/train/4/9555061L.png"
    right_img = "data/raw/archive2/train/4/9555061R.png"
    save_img = "outputs/vis/week6/jsw_medial_final.png"

    compare_jsw(left_img, right_img, save_img, px_to_mm=0.1)
