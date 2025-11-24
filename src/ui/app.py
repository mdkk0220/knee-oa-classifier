import streamlit as st
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import random
from pathlib import Path
import base64
from io import BytesIO

# ============================================================
# 🔹 경로 설정
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
def asset(*parts) -> Path:
    """src/ui/ 기준으로 자원 경로를 Path 객체로 반환"""
    return BASE_DIR.parent.joinpath(*parts)

# ============================================================
# 🔹 페이지 설정
# ============================================================
st.set_page_config(page_title="퇴행성 무릎 관절염 보조시스템", layout="wide")

# 🔹 CSS
st.markdown("""
<style>
h1, h2, h3, p { text-align: center; }
footer { text-align: center; color: gray; margin-top: 40px; }

/* 버튼 스타일 */
button, .stButton>button {
    background-color: #4C84FF;
    color: white;
    border: none;
    padding: 10px 25px;
    font-size: 16px;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s ease;
}
.stButton>button:hover { background-color: #3b6bdd; }

/* ✅ 탭 디자인 (색상 변경 포함) */
.stTabs [data-baseweb="tab-list"] {
    justify-content: center;
}
.stTabs [data-baseweb="tab"] {
    color: #555 !important;
    background-color: #f0f0f5 !important;
    border-radius: 8px 8px 0 0;
    padding: 8px 16px !important;
    margin-right: 6px;
}
.stTabs [aria-selected="true"] {
    color: white !important;
    background-color: #4C84FF !important;
    font-weight: 600;
    border-bottom: 3px solid #0033cc !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background-color: #dfe6ff !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 🔹 세션 상태 관리
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "main"
if "results" not in st.session_state:
    st.session_state.results = {}

# ============================================================
# ✅ 보조 함수들
# ============================================================
def simulate_analysis():
    kl = random.randint(1, 4)
    bone = random.randint(0, 4)
    risk = ["정상", "의심", "경증", "중등도"][kl - 1]
    return kl, bone, risk

def make_variants(image_pil):
    img = np.array(image_pil.convert("RGB"))
    h, w, _ = img.shape

    def overlay_color(base_img, top_color, x1, y1, x2, y2, alpha=0.3):
        overlay = base_img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), top_color, -1)
        return cv2.addWeighted(overlay, alpha, base_img, 1 - alpha, 0)

    full = overlay_color(img, (0, 255, 0), int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9), 0.25)
    joint = overlay_color(img, (255, 255, 0), int(w*0.3), int(h*0.4), int(w*0.7), int(h*0.45), 0.4)
    bone = overlay_color(img, (255, 0, 0), int(w*0.4), int(h*0.6), int(w*0.45), int(h*0.65), 0.5)
    heatmap = cv2.applyColorMap(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLORMAP_JET)

    return {
        "원본": Image.fromarray(img),
        "전체 영역": Image.fromarray(full),
        "관절 간격": Image.fromarray(joint),
        "골극": Image.fromarray(bone),
        "히트맵": Image.fromarray(heatmap)
    }

# ============================================================
# ✅ 메인 페이지
# ============================================================
if st.session_state.page == "main":
    st.markdown("""
    <div style="text-align:center;">
        <h1>퇴행성 무릎 관절염 분류 보조시스템 📊</h1>
        <p>환자들의 퇴행성 무릎 관절염을 더 자세히 판별 도와드립니다!</p>
    </div>
    """, unsafe_allow_html=True)

    # 이미지 경로 설정
    banner_jpg = asset("images", "home_banner.jpg")
    banner_png = asset("images", "home_banner.png")
    banner_path = banner_png if banner_png.exists() else banner_jpg

    # 버튼 중앙 정렬
    col1, col2, col3 = st.columns([4, 6, 4])
    with col2:
        start_clicked = st.button("🚀 시작하기", key="start_button")

    # 배너 이미지 중앙 정렬 + 크기 조정
    if banner_path.exists():
        banner_image = Image.open(banner_path)
        display_width = 650
        banner_image = banner_image.resize((display_width, int(banner_image.height * display_width / banner_image.width)))
        buffered = BytesIO()
        banner_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; align-items: center; margin-top: 20px;">
                <img src="data:image/png;base64,{img_base64}" alt="홈 화면 배너"
                     style="border-radius:12px; width:{display_width}px; box-shadow:0 4px 12px rgba(0,0,0,0.15);">
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠️ 배너 이미지를 찾을 수 없습니다.")

    if start_clicked:
        st.session_state.page = "result_basic"
        st.rerun()

    st.markdown("""
    ---
    <div style="text-align:center;">
        <b>퇴행성 무릎 관절염 분류 보조시스템</b><br>
        2025학년도 2학기 설계 및 기본 프로젝트 심화 6조<br>
        최재하 · 성명규 · 강수아 · 박경빈 · 장미
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# ✅ 결과 페이지 1: 이미지 표시
# ============================================================
elif st.session_state.page == "result_basic":
    st.header("📊 상세 진단 결과")

    col_btn1, col_btn2, col_btn3 = st.columns([1, 8, 1])
    with col_btn1:
        if st.button("🏠 홈화면"):
            st.session_state.page = "main"
            st.rerun()
    with col_btn3:
        if st.button("결과분석"):
            st.session_state.page = "result_detail"
            st.rerun()

    st.markdown("---")

    # ✅ 왼쪽/오른쪽 컬럼 설정
    col_left, col_right = st.columns([1, 1], gap="large")

    # ----------------------------
    # 좌측 무릎
    # ----------------------------
    with col_left:
        st.subheader("좌측 무릎")
        left_file = st.file_uploader("좌측 무릎 이미지 업로드", type=["png", "jpg", "jpeg"], key="left")

        if left_file:
            image_pil = Image.open(left_file)
            st.session_state.left_image = image_pil
            variants = make_variants(image_pil)
            kl, bone, risk = simulate_analysis()
            st.session_state.results["left"] = (kl, bone, risk)
            st.session_state.left_variants = variants

        elif "left_image" in st.session_state:
            variants = st.session_state.left_variants
            kl, bone, risk = st.session_state.results.get("left", (None, None, None))
        else:
            variants = None

        if variants:
            tabs = st.tabs(list(variants.keys()))
            for i, key in enumerate(variants.keys()):
                with tabs[i]:
                    st.image(variants[key], caption=f"좌측 {key}", width=500)

    # ----------------------------
    # 우측 무릎
    # ----------------------------
    with col_right:
        st.subheader("우측 무릎")
        right_file = st.file_uploader("우측 무릎 이미지 업로드", type=["png", "jpg", "jpeg"], key="right")

        if right_file:
            image_pil = Image.open(right_file)
            st.session_state.right_image = image_pil
            variants = make_variants(image_pil)
            kl, bone, risk = simulate_analysis()
            st.session_state.results["right"] = (kl, bone, risk)
            st.session_state.right_variants = variants

        elif "right_image" in st.session_state:
            variants = st.session_state.right_variants
            kl, bone, risk = st.session_state.results.get("right", (None, None, None))
        else:
            variants = None

        if variants:
            tabs = st.tabs(list(variants.keys()))
            for i, key in enumerate(variants.keys()):
                with tabs[i]:
                    st.image(variants[key], caption=f"우측 {key}", width=500)



# ============================================================
# ✅ 결과 페이지 2: 세부 분석 및 그래프
# ============================================================
elif st.session_state.page == "result_detail":
    st.header("📊 상세 진단 결과")

    # 🔹 네 개 컬럼 만들기 (좌우 여백 포함)
    col1, col2, col3, col4 = st.columns([1, 2, 2, 1])

    with col2:
        if st.button("🏠 홈화면", use_container_width=True):
            st.session_state.page = "main"
            st.rerun()

    with col3:
        if st.button("이미지 진단", use_container_width=True):
            st.session_state.page = "result_basic"
            st.rerun()

    st.markdown("---")


    results = st.session_state.results
    if "left" not in results or "right" not in results:
        st.warning("⚠️ 먼저 양쪽 이미지를 업로드해주세요 !")
    else:
        kl_left, bone_left, risk_left = results["left"]
        kl_right, bone_right, risk_right = results["right"]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("좌측 결과")
            st.info(f"KL 등급: {kl_left}")
            st.info(f"골극 개수: {bone_left}")
            st.error(f"위험도: {risk_left}")

        with col2:
            st.subheader("우측 결과")
            st.info(f"KL 등급: {kl_right}")
            st.info(f"골극 개수: {bone_right}")
            st.error(f"위험도: {risk_right}")

        st.markdown("---")
        st.subheader("좌·우 비교 그래프")

        colA, colB = st.columns(2)
        with colA:
            st.markdown("**KL 등급 비교**")
            fig, ax = plt.subplots()
            ax.bar(["좌측", "우측"], [kl_left, kl_right], color=["skyblue", "lightcoral"])
            ax.set_ylim(0, 5)
            st.pyplot(fig)

        with colB:
            st.markdown("**골극 개수 비교**")
            fig2, ax2 = plt.subplots()
            ax2.bar(["좌측", "우측"], [bone_left, bone_right], color=["skyblue", "lightcoral"])
            ax2.set_ylim(0, 5)
            st.pyplot(fig2)
