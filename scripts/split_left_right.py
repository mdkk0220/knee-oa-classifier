# ============================================================
# 📁 scripts/split_left_right.py
# ------------------------------------------------------------
# 단일 무릎 X-ray 한 장에서 왼쪽/오른쪽 무릎을 자동으로 분리해서 저장
# ------------------------------------------------------------
# 사용 예시:
#   python scripts/split_left_right.py --img path/to/xray.png
#   python scripts/split_left_right.py --img path/to/xray.png --out_dir data/split
#
# 동작:
#   1) 흑백 X-ray 로드
#   2) 세로 방향(컬럼) 밝기 프로파일 분석
#   3) 중앙부(35%~65%)에서 가장 밝기가 높은/낮은 지점 중
#      "양 무릎 사이 공간"에 해당하는 최소 밝기 위치를 찾음
#   4) 해당 위치를 기준으로 좌/우 이미지 분리
#   5) *_L.png, *_R.png 이름으로 저장
# ============================================================

import argparse
from pathlib import Path

import cv2
import numpy as np


def find_split_column(gray: np.ndarray) -> int:
    """
    양 무릎 사이의 세로 기준선(column)을 자동으로 찾는 함수.

    아이디어:
      - 각 세로 컬럼별 평균 밝기(0~255)를 계산
      - 중앙부(가로 35% ~ 65%)만 대상으로 함
      - 그 영역에서 평균 밝기가 최소인 컬럼을 '무릎 사이 공간'으로 가정
      - 너무 한쪽으로 치우친 경우에는 width//2 를 fallback 으로 사용
    """
    h, w = gray.shape

    # 세로 방향(각 column)의 평균 intensity 계산
    col_mean = gray.mean(axis=0)  # shape: (W,)

    # 약간 smoothing 해서 노이즈 제거
    k = 31  # 홀수 kernel size
    kernel = np.ones(k) / k
    col_mean_smooth = np.convolve(col_mean, kernel, mode="same")

    # 중앙부(35% ~ 65%)만 검색
    left_idx = int(w * 0.35)
    right_idx = int(w * 0.65)

    mid_region = col_mean_smooth[left_idx:right_idx]
    min_rel_idx = int(mid_region.argmin())
    split_col = left_idx + min_rel_idx

    # 너무 한쪽으로 치우쳐 있으면 안전하게 중앙으로
    if split_col < int(w * 0.25) or split_col > int(w * 0.75):
        split_col = w // 2

    return split_col


def split_left_right(img_path: Path, out_dir: Path | None = None) -> tuple[Path, Path]:
    """
    입력:  한 장의 X-ray 이미지 경로
    출력: 분리된 왼쪽/오른쪽 이미지 경로 (left_path, right_path)
    """
    if out_dir is None:
        out_dir = img_path.parent

    out_dir.mkdir(parents=True, exist_ok=True)

    # 이미지 로드 (흑백)
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {img_path}")

    h, w = gray.shape
    print(f"🖼 원본 이미지 크기: {w} x {h}")

    # 분할 기준 column 찾기
    split_col = find_split_column(gray)
    print(f"✂️ 분할 기준 column: {split_col} / {w}")

    # 좌/우 잘라내기
    left_img = gray[:, :split_col]
    right_img = gray[:, split_col:]

    # 너무 얇게 잘리면 안전하게 중앙 기준으로 재조정
    if left_img.shape[1] < w * 0.2 or right_img.shape[1] < w * 0.2:
        split_col = w // 2
        left_img = gray[:, :split_col]
        right_img = gray[:, split_col:]
        print("⚠️ 한쪽 폭이 너무 작아서 중앙 분할로 fallback 적용")

    # 파일명 생성
    stem = img_path.stem  # 예: "9003175"
    left_path = out_dir / f"{stem}L.png"
    right_path = out_dir / f"{stem}R.png"

    # 저장
    cv2.imwrite(str(left_path), left_img)
    cv2.imwrite(str(right_path), right_img)

    print(f"✅ LEFT  저장: {left_path}")
    print(f"✅ RIGHT 저장: {right_path}")

    return left_path, right_path


def main():
    parser = argparse.ArgumentParser(description="단일 X-ray에서 왼쪽/오른쪽 무릎 자동 분리")
    parser.add_argument("--img", required=True, help="입력 X-ray 이미지 경로")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="분리된 L/R 이미지를 저장할 디렉토리 (기본값: 입력 이미지와 동일 폴더)",
    )
    args = parser.parse_args()

    img_path = Path(args.img)
    if not img_path.exists():
        raise FileNotFoundError(f"입력 이미지가 존재하지 않습니다: {img_path}")

    out_dir = Path(args.out_dir) if args.out_dir else None

    print("✨ 한 장 → 좌/우 자동 분리 시작")
    print(f"  입력 이미지: {img_path}")

    left_path, right_path = split_left_right(img_path, out_dir)

    print("────────────────────────────")
    print("🎯 분리 완료")
    print(f"LEFT  : {left_path}")
    print(f"RIGHT : {right_path}")
    print("이 경로를 그대로 compare_knees_pair_test.py --left/--right 에 넣으면 됨.")
    print("────────────────────────────")


if __name__ == "__main__":
    main()
