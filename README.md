# 1️⃣ 본인 컴퓨터에 복제
cd C:\Users\<이름>\Documents
git clone https://github.com/mdkk0220/knee-oa-classifier.git
cd knee-oa-classifier

# 2️⃣ 가상환경 만들기
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3️⃣ 의존성 설치
pip install -r requirements.txt

# 4️⃣ pre-commit 설치
pre-commit install



#이후부터는 작업 시작할 때마다 이렇게 👇
git pull origin main             # 최신 코드 받아오기
git checkout -b feature/<이름>-<작업명>   # 새 브랜치 생성

# 코드 수정 후
git add .
git commit -m "feat: <작업 설명>"
git push origin feature/<이름>-<작업명>

여기서부터는 우리 시스템 설명
# 🦵 Knee OA Classifier  
퇴행성 무릎관절염 Kellgren–Lawrence 등급 분류 보조 시스템

---

## 📘 프로젝트 개요
X-ray 영상으로 무릎의 퇴행성 관절염 단계를 자동 분류하는 AI 보조 시스템입니다.  
ResNet50 기반 Transfer Learning, Grad-CAM 시각화, Hugging Face UI 통합을 포함합니다.

---

## 🧩 프로젝트 구조

knee-oa-classifier/
├── data/ # 데이터 관련 폴더
│ ├── raw/ # 원본 X-ray
│ ├── interim/ # 임시 처리 데이터
│ └── processed/ # 전처리 완료 데이터
├── src/ # 핵심 소스코드
│ ├── data/ # 데이터 로더
│ ├── models/ # 모델 정의 (ResNet50 등)
│ ├── train/ # 학습 스크립트
│ ├── eval/ # 평가 스크립트
│ ├── explain/ # Grad-CAM 시각화
│ └── ui/ # Gradio / Hugging Face UI
├── notebooks/ # 실험용 Jupyter 노트북
├── reports/ # 결과 및 시각화 저장
├── tests/ # 테스트 코드
└── README.md

yaml
코드 복사

---

## ⚙️ 개발 환경
- Python 3.10+
- PyTorch
- OpenCV / NumPy / Pandas
- Grad-CAM / Matplotlib
- Gradio / Hugging Face

---

## 🧠 주요 담당자

| 이름 | 역할 | 주요 작업 |
|------|------|------------|
| **성명규** | 환경 관리 / 모델 | Git 관리, 구조 설계, ResNet50 구현 |
| **박경빈** | 데이터 전처리 | X-ray 정제, 증강, 불균형 보정 |
| **강수아** | 웹 UI 구현 | Gradio, Hugging Face UI 제작 |
| **장미** | 시스템 테스트 | 통합 검증, 오류 분석 |
| **최재하** | 시각화 / 해석 | Grad-CAM, 결과 분석 |

---


2️⃣ 가상환경 및 의존성 설치
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pre-commit install

3️⃣ 브랜치 생성
git checkout -b feature/<이름>-<작업명>

4️⃣ 작업 후 커밋 & 푸시
git add .
git commit -m "feat: 작업 내용 설명"
git push origin feature/<이름>-<작업명>

5️⃣ Pull Request 생성

GitHub → “Pull Requests” → “New pull request”

base: main / compare: feature/...

리뷰 요청 → 성명규가 merge 승인
