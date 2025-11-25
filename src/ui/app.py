# ============================================================
# src/ui/app.py
# Streamlit UI + ResNet50 BackboneFC + GradCAM 통합
# 좌/우 업로더 + 5탭(원본/전체영역/관절간격/골극/히트맵)
# KL/위험도/골극 + 좌우 비교 그래프 + 결과 수치 카드
# ============================================================

from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

from src.explain.gradcam import GradCAM

# ------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
labels = [f"KL-{i}" for i in range(5)]

MODEL_DIR = Path("outputs/resnet50_finetune_combined")
MODEL_PATH = MODEL_DIR / "model_best.pth"


def asset(*parts) -> Path:
    return BASE_DIR.parent.joinpath(*parts)


# ------------------------------------------------------------
# Streamlit 세션
# ------------------------------------------------------------
st.set_page_config(page_title="퇴행성 무릎 관절염 보조시스템", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "main"

for key in ["results", "left_variants", "right_variants", "model_accuracy"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ------------------------------------------------------------
# CSS (정렬/중앙배치 보정 최종 버전)
# ------------------------------------------------------------
st.markdown(
    """
<style>
h1, h2, h3, p { text-align: center; }

/* 전체 패딩 조금 줄이기 */
.block-container { padding-top: 1rem; }

/* 파일 업로더 중앙 정렬 */
[data-testid="stFileUploader"] {
    max-width: 420px;
    margin-left: auto;
    margin-right: auto;
}

/* 탭 중앙 배치 및 스타일 */
.stTabs [data-baseweb="tab-list"] {
    justify-content: center !important;
}
.stTabs [data-baseweb="tab"] {
    min-width: 90px !important;
    text-align: center !important;
    padding: 6px 20px !important;
    margin-right: 6px;
    color: #555 !important;
    background-color: #f0f0f5 !important;
    border-radius: 8px 8px 0 0;
}
.stTabs [aria-selected="true"] {
    color: white !important;
    background-color: #4C84FF !important;
    font-weight: 600;
}

/* 이미지 중앙 + 그림자 */
.center-img-box {
    display: flex;
    justify-content: center;
    width: 100%;
    margin-top: 12px;
}
.preview-img {
    width: 300px !important;
    height: auto !important;
    border-radius: 10px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
}

/* 결과 카드 중앙 */
.center-card {
    display: flex;
    justify-content: center;
    margin-top: 14px;
}

/* 버튼 스타일 (줄바꿈 방지) */
.stButton>button {
    background-color: #4C84FF;
    color: white;
    border: none;
    padding: 8px 20px;
    font-size: 15px;
    border-radius: 12px;
    cursor: pointer;
    transition: all 0.2s ease;
    white-space: nowrap;          /* 글자 줄바꿈 방지 */
}
.stButton>button:hover { background-color: #3b6bdd; }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# 모델 로딩
# ------------------------------------------------------------
_model: torch.nn.Module | None = None
_model_accuracy: float | None = None


class ResNet50KL_BackboneFC(nn.Module):
    """
    compare_knees_confidence.py와 동일한 구조:
    - self.backbone: resnet50
    - self.backbone.fc: Linear -> BN -> ReLU -> Dropout -> Linear
    """

    def __init__(self, num_classes: int = 5, pretrained: bool = False):
        super().__init__()
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_resnet50_trained(weight_path: str | Path, num_classes: int = 5) -> nn.Module:
    weight_path = str(weight_path)
    ckpt = torch.load(weight_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)

    clean: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        clean[k] = v

    model = ResNet50KL_BackboneFC(num_classes=num_classes, pretrained=False)
    missing, unexpected = model.load_state_dict(clean, strict=False)
    print(f"[UI] Loaded weights from {weight_path}")
    print(f"[UI] Missing: {len(missing)} | Unexpected: {len(unexpected)}")
    if missing:
        print("  Missing:", missing)
    if unexpected:
        print("  Unexpected:", unexpected)

    model.to(device).eval()
    return model


def load_model() -> nn.Module:
    global _model
    if _model is None:
        _model = build_resnet50_trained(MODEL_PATH, num_classes=5)
    return _model

# ------------------------------------------------------------
# 정확도 박스
# ------------------------------------------------------------
def load_model_accuracy() -> float | None:
    global _model_accuracy
    if _model_accuracy is not None:
        return _model_accuracy

    metrics = MODEL_DIR / "metrics.json"
    if metrics.exists():
        j = json.load(open(metrics, "r", encoding="utf-8"))
        if "best_val_acc" in j:
            _model_accuracy = float(j["best_val_acc"])
            return _model_accuracy
    return None


def render_accuracy_box():
    acc = load_model_accuracy()
    if acc is not None:
        st.markdown(
            f"""
            <div style="
                margin: 12px auto 24px auto;
                max-width: 420px;
                padding: 14px 20px;
                border-radius: 14px;
                background: rgba(76,132,255,0.08);
                border: 1px solid rgba(76,132,255,0.35);
                text-align: center;
                font-size: 15px;">
                <div style="font-weight: 600; margin-bottom: 4px; color:#233866;">모델 검증 정확도</div>
                <div style="font-size: 26px; font-weight: 800; color: #4C84FF;">
                    {acc*100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ------------------------------------------------------------
# 전처리 / 예측
# ------------------------------------------------------------
tf = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


def predict_single(img: Image.Image) -> Dict:
    model = load_model()
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0].cpu()

    conf, idx = torch.max(prob, dim=0)
    kl_raw = int(idx.item())
    kl = max(0, min(4, kl_raw))
    return {"kl": kl, "confidence": float(conf.item()), "probs": prob.tolist()}

# ------------------------------------------------------------
# CAM/시각화
# ------------------------------------------------------------
def _overlay_color(base_img_np, color, x1, y1, x2, y2, alpha=0.3):
    overlay = base_img_np.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    return cv2.addWeighted(overlay, alpha, base_img_np, 1 - alpha, 0)


def generate_cam_variants(img: Image.Image):
    """
    - CAM 계산: 원본 이미지를 tf(224) 거쳐서 사용
    - 시각화: 512x512로 리사이즈한 한 장 기준으로 오버레이
    """
    orig = img.convert("RGB")
    base_vis = orig.resize((512, 512))
    base_np = np.array(base_vis)
    h, w, _ = base_np.shape

    full_np = _overlay_color(
        base_np,
        (0, 255, 0),
        int(w * 0.1),
        int(h * 0.1),
        int(w * 0.9),
        int(h * 0.9),
        alpha=0.25,
    )
    joint_np = _overlay_color(
        base_np,
        (255, 255, 0),
        int(w * 0.25),
        int(h * 0.40),
        int(w * 0.75),
        int(h * 0.55),
        alpha=0.40,
    )
    bone_np = _overlay_color(
        base_np,
        (255, 0, 0),
        int(w * 0.40),
        int(h * 0.60),
        int(w * 0.60),
        int(h * 0.80),
        alpha=0.50,
    )

    model = load_model()
    backbone = getattr(model, "backbone", model)
    x = tf(orig).unsqueeze(0).to(device)
    gradcam = GradCAM(backbone, target_layer=backbone.layer4[-1])
    cam, _ = gradcam(x)

    cam_np = cv2.resize(np.clip(cam.cpu().numpy(), 0, 1), (w, h))
    heat = cv2.applyColorMap((cam_np * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    alpha = 0.40 if cam_np.mean() > 0.40 else 0.55
    heat_overlay = (base_np * (1 - alpha) + heat * alpha).astype(np.uint8)

    return (
        base_vis,
        Image.fromarray(full_np),
        Image.fromarray(joint_np),
        Image.fromarray(bone_np),
        Image.fromarray(heat_overlay),
    )

# ------------------------------------------------------------
# KL → 위험도/골극
# ------------------------------------------------------------
def _safe_kl(kl: int) -> int:
    try:
        k = int(kl)
    except Exception:
        k = 0
    return max(0, min(4, k))


def risk_map(kl: int) -> str:
    return ["정상", "의심", "경증", "중등도", "중증"][_safe_kl(kl)]


def bone_map(kl: int) -> int:
    return _safe_kl(kl)

# ------------------------------------------------------------
# 분석
# ------------------------------------------------------------
def analyze_side(img: Image.Image) -> Dict:
    pred = predict_single(img)
    base, full, joint, bone_img, heatmap = generate_cam_variants(img)
    kl = pred["kl"]
    return {
        "kl": kl,
        "confidence": pred["confidence"],
        "risk": risk_map(kl),
        "bone": bone_map(kl),
        "orig": base,
        "full": full,
        "joint": joint,
        "bone_img": bone_img,
        "heatmap": heatmap,
    }


def make_variants(res: Dict):
    return {
        "원본": res["orig"],
        "전체 영역": res["full"],
        "관절 간격": res["joint"],
        "골극": res["bone_img"],
        "히트맵": res["heatmap"],
    }

# ============================================================
# 페이지 1 : 홈
# ============================================================
if st.session_state.page == "main":

    st.markdown(
        """
        <div style="text-align:center;">
            <h1>퇴행성 무릎 관절염 분류 보조시스템 📊</h1>
            <p>환자들의 퇴행성 무릎 관절염을 더 자세히 판별 도와드립니다!</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    banner_jpg = asset("images", "home_banner.jpg")
    banner_png = asset("images", "home_banner.png")
    banner_path = banner_png if banner_png.exists() else banner_jpg

    col1, col2, col3 = st.columns([4, 6, 4])
    with col2:
        clicked = st.button("🚀 시작하기", use_container_width=True)

    if banner_path.exists():
        img = Image.open(banner_path)
        w = 650
        img = img.resize((w, int(img.height * w / img.width)))
        buf = BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        st.markdown(
            f"""
            <div style="display:flex; justify-content:center; margin-top:20px;">
                <img src="data:image/png;base64,{img_b64}"
                     style="border-radius:12px; width:{w}px; box-shadow:0 4px 12px rgba(0,0,0,0.15);"/>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if clicked:
        st.session_state.page = "result_basic"
        st.rerun()

    st.markdown(
        """
        ---  
        <div style="text-align:center;">
            <b>퇴행성 무릎 관절염 분류 보조시스템</b><br>
            2025학년도 2학기 설계 및 기본 프로젝트 심화 6조<br>
            최재하 · 성명규 · 강수아 · 박경빈 · 장미
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# 페이지 2 : 이미지 진단
# ============================================================
elif st.session_state.page == "result_basic":

    # 제목을 버튼과 분리해서 항상 중앙에 고정
    st.markdown(
        """
        <h1 style='text-align:center; margin-top:25px; margin-bottom:5px; font-size:32px;'>
        📊 이미지 기반 진단 결과
        </h1>
        """,
        unsafe_allow_html=True,
    )

    # 상단 버튼 (좌: 홈, 우: 결과분석)
    col_a, col_mid, col_c = st.columns([2, 6, 2])
    with col_a:
        if st.button("🏠 홈 화면", key="btn_home_basic", use_container_width=True):
            st.session_state.page = "main"
            st.rerun()
    with col_c:
        if st.button("결과 분석", key="btn_detail_basic", use_container_width=True):
            st.session_state.page = "result_detail"
            st.rerun()

    st.markdown(
        "<hr style='margin-top:10px; margin-bottom:10px;'>", unsafe_allow_html=True
    )

    render_accuracy_box()

    # 좌우 1:1 균등 분할
    col_left, col_right = st.columns([1, 1], gap="large")

    # -----------------------
    # 좌측
    # -----------------------
    with col_left:
        st.subheader("좌측 무릎")
        left_file = st.file_uploader(
            "좌측 무릎 이미지 업로드", type=["png", "jpg", "jpeg"], key="left"
        )

        if left_file is not None:
            raw = left_file.read()
            img = Image.open(BytesIO(raw)).convert("RGB")
            L = analyze_side(img)
            st.session_state.results = st.session_state.results or {}
            st.session_state.results["left"] = L
            st.session_state.left_variants = make_variants(L)

        variants = st.session_state.left_variants
        if variants:
            tabs = st.tabs(list(variants.keys()))
            for i, key in enumerate(variants.keys()):
                with tabs[i]:
                    # 탭 바로 아래 X-ray 중앙 정렬 + 폭 300
                    st.markdown('<div class="center-img-box">', unsafe_allow_html=True)
                    st.image(
                        variants[key],
                        output_format="PNG",
                        width=300,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                    if st.session_state.results and "left" in st.session_state.results:
                        L = st.session_state.results["left"]
                        st.markdown(
                            f"""
                            <div class="center-card">
                              <div style="
                                  padding: 14px 20px; border-radius: 12px;
                                  background: rgba(76,132,255,0.08);
                                  border: 1px solid rgba(76,132,255,0.25);
                                  max-width: 380px;
                                  text-align:center;">
                                <b>KL 등급:</b> {L['kl']}<br>
                                <b>확신도:</b> {L['confidence']:.2f}<br>
                                <b>위험도:</b> {L['risk']}<br>
                                <b>골극 개수:</b> {L['bone']}
                              </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
        else:
            st.info("좌측 이미지를 아직 분석하지 않았습니다.")

    # -----------------------
    # 우측
    # -----------------------
    with col_right:
        st.subheader("우측 무릎")
        right_file = st.file_uploader(
            "우측 무릎 이미지 업로드", type=["png", "jpg", "jpeg"], key="right"
        )

        if right_file is not None:
            raw = right_file.read()
            img = Image.open(BytesIO(raw)).convert("RGB")
            R = analyze_side(img)
            st.session_state.results = st.session_state.results or {}
            st.session_state.results["right"] = R
            st.session_state.right_variants = make_variants(R)

        variants = st.session_state.right_variants
        if variants:
            tabs = st.tabs(list(variants.keys()))
            for i, key in enumerate(variants.keys()):
                with tabs[i]:
                    st.markdown('<div class="center-img-box">', unsafe_allow_html=True)
                    st.image(
                        variants[key],
                        output_format="PNG",
                        width=300,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                    if st.session_state.results and "right" in st.session_state.results:
                        R = st.session_state.results["right"]
                        st.markdown(
                            f"""
                            <div class="center-card">
                              <div style="
                                  padding: 14px 20px; border-radius: 12px;
                                  background: rgba(76,132,255,0.08);
                                  border: 1px solid rgba(76,132,255,0.25);
                                  max-width: 380px;
                                  text-align:center;">
                                <b>KL 등급:</b> {R['kl']}<br>
                                <b>확신도:</b> {R['confidence']:.2f}<br>
                                <b>위험도:</b> {R['risk']}<br>
                                <b>골극 개수:</b> {R['bone']}
                              </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
        else:
            st.info("우측 이미지를 아직 분석하지 않았습니다.")

# ============================================================
# 페이지 3 : 상세 수치
# ============================================================
elif st.session_state.page == "result_detail":

    st.markdown(
        """
        <h1 style='text-align:center; margin-top:25px; margin-bottom:5px; font-size:32px;'>
        📈 수치 기반 상세 진단 결과
        </h1>
        """,
        unsafe_allow_html=True,
    )

    col1, col_mid, col4 = st.columns([2, 6, 2])
    with col1:
        if st.button("🏠 홈 화면", key="btn_home_detail", use_container_width=True):
            st.session_state.page = "main"
            st.rerun()

    with col4:
        if st.button("이미지 진단", key="btn_basic_detail", use_container_width=True):
            st.session_state.page = "result_basic"
            st.rerun()

    st.markdown(
        "<hr style='margin-top:10px; margin-bottom:10px;'>", unsafe_allow_html=True
    )

    render_accuracy_box()

    results = st.session_state.results
    if not results or "left" not in results or "right" not in results:
        st.warning("⚠️ 먼저 이미지 페이지에서 X-ray를 업로드하고 분석을 실행해주세요.")
    else:
        L = results["left"]
        R = results["right"]

        colL, colR = st.columns(2)
        with colL:
            st.subheader("좌측 결과")
            st.info(f"KL 등급: {L['kl']}")
            st.info(f"골극 개수: {L['bone']}")
            st.error(f"위험도: {L['risk']}")

        with colR:
            st.subheader("우측 결과")
            st.info(f"KL 등급: {R['kl']}")
            st.info(f"골극 개수: {R['bone']}")
            st.error(f"위험도: {R['risk']}")

        st.markdown("---")
        st.subheader("좌·우 비교 그래프")

        colA, colB = st.columns(2)
        with colA:
            fig, ax = plt.subplots()
            ax.bar(["좌측", "우측"], [L["kl"], R["kl"]])
            ax.set_ylim(0, 5)
            ax.set_ylabel("KL Grade")
            st.pyplot(fig)

        with colB:
            fig2, ax2 = plt.subplots()
            ax2.bar(["좌측", "우측"], [L["bone"], R["bone"]])
            ax2.set_ylim(0, 5)
            ax2.set_ylabel("Bone Spurs (count)")
            st.pyplot(fig2)
