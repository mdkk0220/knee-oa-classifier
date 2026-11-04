# 팀 구성과 협업 규칙

## 👥 팀 구성과 역할

| 이름 | 역할 | 주요 작업 |
|---|---|---|
| **성명규** | 환경 관리 · 모델 | 저장소 구조 설계, Git 협업 규칙 운영, ResNet50 구현 |
| 박경빈 | 데이터 전처리 | X-ray 정제, 증강, 클래스 불균형 보정 |
| 강수아 | 웹 UI | Gradio, Hugging Face UI 제작 |
| 장미 | 시스템 테스트 | 통합 검증, 오류 분석 |
| 최재하 | 시각화 · 해석 | Grad-CAM, 결과 분석 |

<details>
<summary>🤝 팀 협업 규칙 (팀원용)</summary>

```bash
git pull origin main                          # 최신 코드 받기
git checkout -b feature/<이름>-<작업명>        # 작업 브랜치 만들기
git add .
git commit -m "feat: <작업 설명>"
git push origin feature/<이름>-<작업명>
```

GitHub에서 Pull Request를 만들고(base: `main`), 리뷰 후 성명규가 merge를 승인합니다.
커밋 전에 `pre-commit install`로 자동 검사를 켜 두세요.

</details>

[← 처음으로](../README.md)
