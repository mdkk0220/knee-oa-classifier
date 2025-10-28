# scripts/cam_demo.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import numpy as np
from src.explain.gradcam import GradCAM
from src.explain.viz_utils import overlay_heatmap, save_fig_grid

def build_resnet50_imagenet():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    return model

TR = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

def main():
    img_path = "data/processed/train/2/9000296R.png"  # 예시
    out_path = "outputs/vis/week2/cam_sample.png"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        img = Image.open(img_path).convert("RGB")
    except FileNotFoundError:
        print(f"❌ 이미지 파일을 찾을 수 없습니다: {img_path}")
        return

    x = TR(img).unsqueeze(0)

    model = build_resnet50_imagenet()
    target_layer = model.layer4[-1]
    cam = GradCAM(model, target_layer)
    cam_map, pred_cls = cam(x)

    base = np.array(img.resize((224, 224)))
    overlay = overlay_heatmap(base, cam_map.numpy())

    save_fig_grid(
        [base, (cam_map.numpy() * 255).astype(np.uint8), overlay],
        ["Input (224)", "Grad-CAM (gray)", "Overlay"],
        out_path
    )

    print(f"✅ Saved: {out_path}")
    print(f"✅ Predicted Class Index: {pred_cls}")

if __name__ == "__main__":
    main()
