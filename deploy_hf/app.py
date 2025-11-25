import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
import numpy as np
import cv2
import gradio as gr
from PIL import Image

# ---------------------------------------
# 1) 모델 로드
# ---------------------------------------
MODEL_PATH = "model/model_best.pth"

def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)  # KL 0~4
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model()

# ---------------------------------------
# 2) 전처리
# ---------------------------------------
transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def preprocess(img):
    img = img.convert("RGB")
    return transform(img).unsqueeze(0)

# ---------------------------------------
# 3) 위험도 매핑
# ---------------------------------------
def risk_map(kl):
    mapping = ["정상", "의심", "경증", "중등도", "중증"]
    return mapping[kl]

# ---------------------------------------
# 4) 모델 예측
# ---------------------------------------
def predict(img):
    tensor = preprocess(img)
    with torch.no_grad():
        out = model(tensor)
        prob = torch.softmax(out, dim=1).numpy()[0]
        kl = int(np.argmax(prob))
    return kl, float(prob[kl])

# ---------------------------------------
# 5) 좌우 무릎 비교
# ---------------------------------------
def compare(left_img, right_img):
    if left_img is None or right_img is None:
        return "좌/우 이미지 모두 업로드해주세요.", None, None, None, None

    left_kl, left_conf = predict(left_img)
    right_kl, right_conf = predict(right_img)

    left_risk = risk_map(left_kl)
    right_risk = risk_map(right_kl)

    result_text = f"""
🦵 좌우 무릎 KL 비교 결과

🔹 LEFT  
 - KL 등급: {left_kl}  
 - 신뢰도: {left_conf:.2f}  
 - 상태: {left_risk}

🔹 RIGHT  
 - KL 등급: {right_kl}  
 - 신뢰도: {right_conf:.2f}  
 - 상태: {right_risk}
"""

    return (
        result_text,
        left_kl, left_conf,
        right_kl, right_conf
    )

# ---------------------------------------
# 6) Gradio UI
# ---------------------------------------
with gr.Blocks(title="Knee OA Classifier") as demo:
    gr.Markdown("## 🦵 무릎 KL 등급 자동 판독기 (좌우 비교)")

    with gr.Row():
        left_input = gr.Image(type="pil", label="Left Knee X-ray")
        right_input = gr.Image(type="pil", label="Right Knee X-ray")

    run_btn = gr.Button("좌우 비교 실행")

    result_box = gr.Textbox(label="결과", lines=10)

    with gr.Row():
        left_kl_out = gr.Number(label="Left KL")
        left_conf_out = gr.Number(label="Left Confidence")
        right_kl_out = gr.Number(label="Right KL")
        right_conf_out = gr.Number(label="Right Confidence")

    run_btn.click(
        fn=compare,
        inputs=[left_input, right_input],
        outputs=[result_box, left_kl_out, left_conf_out, right_kl_out, right_conf_out]
    )

demo.launch()
