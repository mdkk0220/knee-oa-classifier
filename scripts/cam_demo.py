# ============================================================
# scripts/cam_demo.py | v3.4 (웹페이지용 단일 출력)
# ------------------------------------------------------------
# ✅ 목적:
#   - Grad-CAM Overlay만 생성 (웹페이지 표시용)
# ------------------------------------------------------------
# ⚙️ 실행:
#   python3 scripts/cam_demo.py
# ============================================================
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import numpy as np
import cv2

# 로컬 모듈
from src.explain.gradcam import GradCAM

# ------------------------------------------------------------
# ✅ 체크포인트에 맞는 모델 정의
# ------------------------------------------------------------
class ResNet50KLCheckpoint(nn.Module):
    """체크포인트 구조에 맞춘 ResNet50KL"""
    def __init__(self, num_classes=5):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ------------------------------------------------------------
# ✅ 학습된 모델 로드 함수
# ------------------------------------------------------------
def build_resnet50_trained(weight_path: str):
    model = ResNet50KLCheckpoint(num_classes=5)
    state_dict = torch.load(weight_path, map_location="cpu")
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("backbone.fc"):
            new_key = k.replace("backbone.fc", "fc")
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

# ------------------------------------------------------------
# ✅ 변환 정의
# ------------------------------------------------------------
TR = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

# ------------------------------------------------------------
# ✅ 히트맵 오버레이 생성
# ------------------------------------------------------------
def create_gradcam_overlay(image, heatmap, alpha=0.5):
    """
    Grad-CAM 오버레이 생성 (웹페이지 표시용)
    
    Args:
        image: 원본 이미지 (H, W, 3)
        heatmap: Grad-CAM 히트맵 (H, W) [0, 1]
        alpha: 투명도 (기본 0.5)
    
    Returns:
        오버레이 이미지 (numpy array)
    """
    # 히트맵을 컬러맵으로 변환
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # 그레이스케일이면 RGB로 변환
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # 크기 맞추기
    if image.shape[:2] != heatmap_colored.shape[:2]:
        heatmap_colored = cv2.resize(heatmap_colored, 
                                     (image.shape[1], image.shape[0]))
    
    # 오버레이
    overlay = (alpha * heatmap_colored + (1 - alpha) * image).astype(np.uint8)
    
    return overlay

# ------------------------------------------------------------
# ✅ 실행 함수
# ------------------------------------------------------------
def main():
    # 🔹 설정
    img_path = "data/raw/archive2/train/2/9429009L.png"
    model_path = "outputs/resnet50_finetune_combined/model_best.pth"
    output_path = "outputs/vis/week3/gradcam_overlay.png"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 🔹 이미지 로드
    try:
        img = Image.open(img_path).convert("RGB")
    except FileNotFoundError:
        print(f"❌ 이미지를 찾을 수 없습니다: {img_path}")
        return
    
    original_img = np.array(img.resize((224, 224)))
    x = TR(img).unsqueeze(0)
    
    # 🔹 모델 로드
    print("⏳ Loading model...")
    model = build_resnet50_trained(model_path)
    print("✅ Model loaded")
    
    # 🔹 Grad-CAM 실행
    print("⏳ Generating Grad-CAM...")
    resnet_layers = list(model.backbone.children())
    target_layer = resnet_layers[7]  # layer4
    
    cam = GradCAM(model, target_layer)
    cam_map, pred_cls = cam(x)
    
    print(f"✅ Grad-CAM generated")
    print(f"   - Predicted KL Grade: {pred_cls}")
    print(f"   - Heatmap range: [{cam_map.min():.3f}, {cam_map.max():.3f}]")
    
    # 🔹 오버레이 생성 및 저장
    heatmap_resized = cv2.resize(cam_map.numpy(), (224, 224))
    overlay = create_gradcam_overlay(original_img, heatmap_resized, alpha=0.4)
    
    # PIL Image로 변환하여 저장
    overlay_pil = Image.fromarray(overlay)
    overlay_pil.save(output_path)
    
    print(f"✅ Saved: {output_path}")
    print(f"\n💡 웹페이지에 표시할 이미지: {output_path}")

if __name__ == "__main__":
    main()