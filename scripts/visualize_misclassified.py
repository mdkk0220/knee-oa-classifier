# ============================================================
# 📁 scripts/visualize_misclassified.py
# 오분류 케이스 자동 시각화 (6주차 고도화)
# ============================================================
import torch
import os
from src.data.dataset import XrayDataset
from src.models.resnet50 import build_resnet50
from src.explain.gradcam import generate_gradcam
from src.explain.viz_utils import read_gray_as_rgb, save_fig_grid
from torch.utils.data import DataLoader

# 모델 로드
model_path = "outputs/resnet50_finetune_combined/model_best.pth"
model = build_resnet50(num_classes=5)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# 데이터 로드
dataset = XrayDataset(split="test")
loader = DataLoader(dataset, batch_size=1, shuffle=False)

save_dir = "outputs/vis/week6_misclassified"
os.makedirs(save_dir, exist_ok=True)

# 오분류만 시각화
for i, (img, label) in enumerate(loader):
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0, pred].item()

    if pred != label.item():  # 오분류 케이스
        img_np = read_gray_as_rgb(dataset.image_paths[i])
        cam = generate_gradcam(model, img_np)
        title = f"GT:{label.item()} → Pred:{pred} ({conf:.2f})"
        save_path = f"{save_dir}/case_{i:04d}.png"
        save_fig_grid([img_np, cam], ["원본", title], save_path)
        print(f"❌ Misclassified saved: {save_path}")
