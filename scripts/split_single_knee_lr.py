# ============================================================
# 📁 scripts/split_single_knee_lr.py
# ------------------------------------------------------------
#  단일 무릎 X-ray(양쪽이 한 장에 있는 이미지) → L/R 자동 분리
#  - 입력: --img "원본 X-ray 경로" (양쪽 무릎이 함께 있는 정면 AP 이미지)
#  - 출력: out_dir 아래에 *_L.png, *_R.png 두 장 저장
#  - 이후 compare_knees_pair_test.py 에 바로 물려서 좌/우 비교 가능
# ============================================================

import os
import argparse
from pathlib import Path

import cv2
import numpy as np


def find_vertical_split(gray: np.ndarray) -> int:
    """
    한 장의 무릎 X-ray에서 좌/우를 나누는 세로 위치를 추정.
    - 가운데 30~70% 구간에서 세로 방향 밝기 평균을 보고,
      '무릎 사이 간격(틈)'을 찾는 방식.
    - 데이터 특성상 완벽하진 않지만, 대부분의 정면 AP 사진에서
      중앙부 틈(검거나 밝기 변화)이 존재한다는 가정.
    """

    h, w = gray.shape
    # 중앙 40% 구간만 사용 (가장자리 노이즈 제거)
    x_start = int(w * 0.3)
    x_end = int(w * 0.7)

    # 각 세로열(컬럼)마다 평균 밝기 계산
    col_mean = gray[:, x_start:x_end].mean(axis=0)

    # 무릎 사이 간격이 상대적으로 "어둡다"는 가정 → 최소값 위치 선택
    # 만약 데이터 특성에 따라 반대라면 argmax로 바꿀 수 있음.
    rel_idx = int(np.argmin(col_mean))
    split_x = x_start + rel_idx

    return split_x


def split_lr(img_path: str, out_dir: str) -> tuple[str, str]:
    """
    단일 X-ray를 불러와 좌/우 이미지로 분리하고 저장.
    반환값: (left_path, right_path)
    """
    img_path = Path(img_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 이미지 로드 (그레이스케일)
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {img_path}")

    h, w = gray.shape

    # 세로 분할 위치 추정
    split_x = find_vertical_split(gray)

    # 안정성 위해 최소/최대 범위 클램프
    split_x = max(int(w * 0.25), min(split_x, int(w * 0.75)))

    # 좌/우 슬라이스
    left = gray[:, :split_x]
    right = gray[:, split_x:]

    # PNG로 저장 (이후 파이프라인과 호환되도록)
    base = img_path.stem
    left_path = out_dir / f"{base}_L.png"
    right_path = out_dir / f"{base}_R.png"

    cv2.imwrite(str(left_path), left)
    cv2.imwrite(str(right_path), right)

    return str(left_path), str(right_path)


def main():
    parser = argparse.ArgumentParser(description="Single X-ray → L/R 자동 분리")
    parser.add_argument(
        "--img",
        required=True,
        help="양쪽 무릎이 함께 있는 X-ray 이미지 경로",
    )
    parser.add_argument(
        "--out_dir",
        default="data/single_split",
        help="분리된 L/R 이미지를 저장할 폴더 (기본: data/single_split)",
    )
    args = parser.parse_args()

    left_path, right_path = split_lr(args.img, args.out_dir)

    print("────────────────────────────")
    print("✅ L/R 자동 분리 완료")
    print(f"LEFT  저장 경로: {left_path}")
    print(f"RIGHT 저장 경로: {right_path}")
    print("────────────────────────────")
    print("이제 이 두 경로를 사용해서 좌/우 비교 스크립트에 넣으면 된다.")
    print("예시:")
    print(
        f"  python scripts/compare_knees_pair_test.py "
        f"--left \"{left_path}\" --right \"{right_path}\""
    )


if __name__ == "__main__":
    main()
