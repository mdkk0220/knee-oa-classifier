# 실행 방법 · 폴더 구조 · 데이터

## ▶️ 실행 방법

```bash
git clone https://github.com/mdkk0220/knee-oa-classifier.git
cd knee-oa-classifier
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**데모 화면 실행** (학습된 모델 포함)
```bash
cd deploy_hf
pip install -r requirements.txt
python app.py
```

**이미지 한 장 예측**
```bash
python scripts/predict_single_knee.py \
  --img 경로/무릎사진.png \
  --weights outputs/resnet50_finetune_combined/model_best.pth
```

## 📂 폴더 구조

```
knee-oa-classifier/
├── src/
│   ├── config/      # 학습 · 추론 설정 (yaml)
│   ├── data/        # 데이터셋 로더, 전처리
│   ├── models/      # ResNet50 정의
│   ├── train/       # 학습 루프, 지표
│   ├── eval/        # 평가
│   ├── explain/     # Grad-CAM
│   ├── inference/   # 추론 파이프라인
│   └── ui/          # Gradio 화면
├── scripts/         # 데이터 재구성, 단일 이미지 예측, Grad-CAM 일괄 생성 등
├── deploy_hf/       # Hugging Face 배포용 앱
├── outputs/         # 학습 로그, 평가 보고서, 시각화 결과
└── data/            # (아래 '데이터' 참고)
```

## 🗂️ 데이터

- **출처:** Chen, P. (2018). *Knee Osteoarthritis Severity Grading Dataset* (Version 1). Mendeley Data. https://doi.org/10.17632/56rmx5bjcr.1
- **원 데이터:** Osteoarthritis Initiative (OAI)
- **라이선스:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **원본 분할:** 학습 5,778장 · 검증 826장 · 테스트 1,656장 · 자동 검출 테스트 1,526장 (`data/raw/archive2/`)
- 프로젝트에서는 환자 · 좌우 무릎 단위로 다시 정리해 사용했습니다 (`scripts/restructure_*.py`). 최종 평가는 테스트 3,052장 기준입니다.

**보조 데이터 (`data/raw/digital_knee_xray/`)**
- **출처:** Gornale, S., Patravali, P. (2020). *Digital Knee X-ray Images* (Version 1). Mendeley Data. https://doi.org/10.17632/t9ndx37v5h.1
- **구성:** 무릎 X-ray 1,650장을 전문의 2명(MedicalExpert-I · II)이 각각 KL 0~4로 판정 → 3,300장
- 라이선스와 이용 조건은 위 Mendeley 페이지 표기를 따릅니다. 최종 평가에는 쓰지 않았습니다.

[← 처음으로](../README.md)
