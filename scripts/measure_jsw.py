# ============================================================
# measure_jsw.py | Medial JSW 측정 (다중 컬럼 평균, 발표용 안정 버전)
# ------------------------------------------------------------
# - 입력: 좌/우 무릎 AP X-ray (각각 한 장씩)
# - 내측(medial) ROI에서 여러 세로 컬럼을 훑어서
#   femur 하연 ~ tibia 상연 간격(px)을 측정
# - 좁은 구간들(하위 20%)만 평균해 대표 JSW로 사용
# - 결과: 각 무릎에 짧은 가로선 + 점선 세로선 + JSW(mm) 텍스트
# - KL 모델 사용 없음 (순수 JSW 측정용)
# ============================================================

import cv2
import numpy as np
from pathlib import Path


# ------------------------------------------------------------
# 1. 내측(medial) ROI 정의
# ------------------------------------------------------------
def get_medial_roi(h: int, w: int, side: str):
    """
    h, w : 단일 무릎 이미지 높이/너비
    side : 'L' (왼쪽 무릎), 'R' (오른쪽 무릎)
    return: (x1, x2, y1, y2)
    """
    # 세로: 관절 부위 중심부만 사용
    y1 = int(h * 0.30)
    y2 = int(h * 0.80)

    side = side.upper()
    if side == "L":
        # 왼쪽 무릎 → 오른쪽이 내측
        x1 = int(w * 0.55)
        x2 = int(w * 0.95)
    else:
        # 오른쪽 무릎 → 왼쪽이 내측
        x1 = int(w * 0.05)
        x2 = int(w * 0.45)

    return x1, x2, y1, y2


# ------------------------------------------------------------
# 2. 한 컬럼에서 femur/tibia 경계 찾기
# ------------------------------------------------------------
def find_pair_on_column(col: np.ndarray, y_min: int, y_max: int, min_px: int, max_px: int):
    """
    col : (H,) 형태 intensity (float32)
    y_min, y_max : 탐색 구간
    min_px, max_px : 허용 JSW px 범위

    return: (dist, top_idx, bottom_idx) 또는 None
    """
    H = col.shape[0]

    # 1D 가우시안 블러 후 gradient
    col_blur = cv2.GaussianBlur(col.reshape(-1, 1), (5, 5), 0).flatten()
    grad = np.abs(np.diff(col_blur))

    y_min = max(0, y_min)
    y_max = min(H - 2, y_max)

    if y_max <= y_min + 2:
        return None

    # 상위 몇 개 gradient 피크 후보
    local = grad[y_min:y_max]
    if local.size == 0:
        return None

    num_peaks = 10
    idx_sorted = np.argsort(local)[::-1]
    idx_sorted = idx_sorted[: num_peaks * 2]
    cand_idx = np.unique(idx_sorted + y_min)
    cand_idx = np.sort(cand_idx)

    if cand_idx.size < 2:
        return None

    center = (y_min + y_max) / 2.0
    best = None
    best_score = 1e9

    for i in range(len(cand_idx)):
        for j in range(i + 1, len(cand_idx)):
            top = int(cand_idx[i])
            bottom = int(cand_idx[j])
            dist = bottom - top
            if dist < min_px or dist > max_px:
                continue

            mid = 0.5 * (top + bottom)

            # 중앙에 가까울수록 + 너무 두꺼운 간격은 패널티
            score = abs(mid - center) + 0.15 * dist

            if score < best_score:
                best_score = score
                best = (dist, top, bottom)

    return best


# ------------------------------------------------------------
# 3. JSW 계산 (다중 컬럼 + 하위 20% 평균)
# ------------------------------------------------------------
def compute_jsw(gray: np.ndarray, side: str, px_to_mm: float = 0.1):
    """
    한 무릎 grayscale 이미지에서 medial JSW(mm)를 계산.
    - 내측 ROI 안 여러 컬럼에서 (dist, x, y_top, y_bottom) 후보 수집
    - 좁은 값들(하위 20%)만 골라 평균 위치/간격 산출
    """
    h, w = gray.shape
    x1, x2, y1, y2 = get_medial_roi(h, w, side)

    roi = gray[y1:y2, x1:x2]
    H, W = roi.shape

    # 세로 탐색 범위 (ROI 안에서)
    y_min = int(H * 0.15)
    y_max = int(H * 0.85)

    # 허용 JSW(px) 범위
    min_px = int(H * 0.03)   # 너무 얇은 노이즈 제거
    max_px = int(H * 0.30)   # 너무 두꺼운 간격 제거

    # 전처리 (대비 향상)
    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)

    candidates = []

    # 컬럼을 일정 간격으로 스캔 (너무 촘촘하면 노이즈 많아짐)
    step = max(1, W // 40)  # 대략 30~40개 컬럼 정도 보도록
    for c in range(0, W, step):
        col = enhanced[:, c].astype(np.float32)
        pair = find_pair_on_column(col, y_min, y_max, min_px, max_px)
        if pair is None:
            continue
        dist, top, bottom = pair

        y_top = y1 + top
        y_bottom = y1 + bottom
        x_mid = x1 + c

        candidates.append((dist, x_mid, y_top, y_bottom))

    # ---------- 후보 없으면 fallback (관절 대충 중앙) ----------
    if not candidates:
        mid_y = int(h * 0.53)
        span = int(h * 0.06)
        y_top = mid_y - span // 2
        y_bottom = mid_y + span // 2
        x_mid = (x1 + x2) // 2
        dist_px = y_bottom - y_top
        jsw_mm = float(np.clip(dist_px * px_to_mm, 0.5, 5.0))
        return jsw_mm, x_mid, y_top, y_bottom

    # ---------- 하위 20% 좁은 간격들만 평균 ----------
    dists = np.array([c[0] for c in candidates], dtype=np.float32)
    q20 = np.percentile(dists, 20)

    good = [c for c in candidates if c[0] <= q20 * 1.2]
    if len(good) < 3:
        good = candidates  # 너무 적으면 전체 사용

    dists_g = np.array([c[0] for c in good], dtype=np.float32)
    xs = np.array([c[1] for c in good], dtype=np.float32)
    tops = np.array([c[2] for c in good], dtype=np.float32)
    bots = np.array([c[3] for c in good], dtype=np.float32)

    dist_px = float(dists_g.mean())
    x_mid = int(xs.mean())
    y_top = int(tops.mean())
    y_bottom = int(bots.mean())

    jsw_mm_raw = dist_px * px_to_mm
    jsw_mm = float(np.clip(jsw_mm_raw, 0.5, 5.0))  # 발표용 범위 클리핑

    return jsw_mm, x_mid, y_top, y_bottom


# ------------------------------------------------------------
# 4. 텍스트 안전하게 그리기 (프레임 안으로 클램프)
# ------------------------------------------------------------
def safe_put_text(img, text, center_x, y, font_scale=0.9, color=(255, 255, 255), thickness=2):
    h, w, _ = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

    margin = 10
    x = int(center_x - tw / 2)
    if x < margin:
        x = margin
    if x + tw > w - margin:
        x = w - margin - tw
    y = max(th + margin, y)

    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


# ------------------------------------------------------------
# 5. 병원 스타일 오버레이 (가로선 + 점선)
# ------------------------------------------------------------
def draw_jsw_overlay(img_bgr, x_mid, y_top, y_bottom, jsw_mm, side):
    overlay = img_bgr.copy()
    h, w, _ = overlay.shape

    line_color = (255, 220, 120)  # 밝은 청록 느낌 (BGR)
    text_color = (255, 255, 255)

    # 짧은 가로선
    line_len = int(w * 0.13)
    x1 = x_mid - line_len // 2
    x2 = x_mid + line_len // 2

    cv2.line(overlay, (x1, y_top), (x2, y_top), line_color, 3, cv2.LINE_AA)
    cv2.line(overlay, (x1, y_bottom), (x2, y_bottom), line_color, 3, cv2.LINE_AA)

    # 점선 세로선
    dash, gap = 6, 6
    y = y_top
    while y < y_bottom:
        y2 = min(y + dash, y_bottom)
        cv2.line(overlay, (x_mid, y), (x_mid, y2), line_color, 2, cv2.LINE_AA)
        y += dash + gap

    # 텍스트: 선 위쪽, 좌/우 영역 안에만
    label = f"{jsw_mm:.2f} mm"
    y_text = max(y_top - 20, 30)

    side = side.upper()
    if side == "L":
        center_x = int(w * 0.25)
    else:
        center_x = int(w * 0.75)

    safe_put_text(overlay, label, center_x, y_text,
                  font_scale=0.9, color=text_color, thickness=2)

    return overlay


# ------------------------------------------------------------
# 6. 단일 무릎 처리
# ------------------------------------------------------------
def process_single(path: str, side: str, px_to_mm: float = 0.1):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    jsw_mm, x_mid, y_top, y_bottom = compute_jsw(gray, side, px_to_mm)
    overlay = draw_jsw_overlay(img, x_mid, y_top, y_bottom, jsw_mm, side)

    return jsw_mm, overlay


# ------------------------------------------------------------
# 7. 좌우 합치기
# ------------------------------------------------------------
def compare_jsw(left_path: str, right_path: str, save_path: str, px_to_mm: float = 0.1):
    L_jsw, L_img = process_single(left_path, "L", px_to_mm)
    R_jsw, R_img = process_single(right_path, "R", px_to_mm)

    h = min(L_img.shape[0], R_img.shape[0])
    L_res = cv2.resize(L_img, (int(L_img.shape[1] * h / L_img.shape[0]), h))
    R_res = cv2.resize(R_img, (int(R_img.shape[1] * h / R_img.shape[0]), h))

    combined = np.hstack([L_res, R_res])

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, combined)

    print(f"Left  Medial JSW ≈ {L_jsw:.2f} mm")
    print(f"Right Medial JSW ≈ {R_jsw:.2f} mm")
    print("저장:", save_path)


# ------------------------------------------------------------
# 8. 실행부 (archive2/test 기준)
# ------------------------------------------------------------
if __name__ == "__main__":
    left_img = "data/raw/archive2/test/1/9008934L.png"
    right_img = "data/raw/archive2/test/1/9008934R.png"
    save_img = "outputs/vis/week7/jsw_medial_multi_column.png"

    compare_jsw(left_img, right_img, save_img, px_to_mm=0.1)
