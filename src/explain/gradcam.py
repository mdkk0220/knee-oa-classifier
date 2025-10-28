# src/explain/gradcam.py

import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision import transforms as T
from src.models.resnet50 import ResNet50KL


def generate_cam(model, image_tensor, target_layer):
    """Grad-CAM 계산"""
    acts, grads = [], []

    def fwd_hook(_, __, output):
        acts.append(output)

    def bwd_hook(_, grad_in, grad_out):
        grads.append(grad_out[0])

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    logits = model(image_tensor)
    cls = logits.argmax(1).item()
    score = logits[0, cls]
    score.backward()

    A = acts[0].detach().cpu().numpy()[0]
    G = grads[0].detach().cpu().numpy()[0]
    weights = G.mean(axis=(1, 2))
    cam = np.maximum((weights[:, None, None] * A).sum(axis=0), 0)
    cam = cv2.resize(cam, (image_tensor.shape[3], image_tensor.shape[2]))
    cam = (cam - cam.min()) / (cam.max() + 1e-6)

    h1.remove()
    h2.remove()
    return cam


def overlay_heatmap(img, cam, alpha=0.4):
    """원본 이미지 위에 heatmap을 겹쳐 시각화"""
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = (alpha * heatmap + (1 - alpha) * img).astype(np.uint8)
    return overlay


def main(weights, input_path, output_path="reports/figures/cam.png", input_size=512):
    """Grad-CAM 실행 메인 함수"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet50KL().to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    target_layer = model.backbone.layer4[-1].conv3  # 마지막 conv layer

    # 입력 전처리
    pil_tf = T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img0 = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)
    x = pil_tf(T.ToPILImage()(img0)).unsqueeze(0).to(device)

    cam = generate_cam(model, x, target_layer)
    overlay = overlay_heatmap(img0, cam)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, overlay)
    print(f"✅ Saved Grad-CAM visualization -> {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Grad-CAM 시각화 실행")
    ap.add_argument("--weights", required=True, help="모델 가중치 경로 (.pth)")
    ap.add_argument("--input", required=True, help="입력 X-ray 이미지 경로")
    ap.add_argument("--output", default="reports/figures/cam.png", help="출력 이미지 경로")
    ap.add_argument("--size", type=int, default=512, help="입력 이미지 크기")
    args = ap.parse_args()

    main(args.weights, args.input, args.output, args.size)
