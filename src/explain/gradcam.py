import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms as T

from src.models.resnet50 import ResNet50KL

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--input", required=True)
ap.add_argument("--output", default="reports/figures/cam.png")
args = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet50KL().to(device)
model.load_state_dict(torch.load(args.weights, map_location=device))
model.eval()

target_layer = model.backbone.layer4[-1].conv3

pil_tf = T.Compose(
    [
        T.Grayscale(num_output_channels=3),
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

img0 = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)
x = pil_tf(T.ToPILImage()(img0)).unsqueeze(0).to(device)

acts, grads = [], []


def fwd_hook(_, __, output):
    acts.append(output)


def bwd_hook(_, grad_in, grad_out):
    grads.append(grad_out[0])


h1 = target_layer.register_forward_hook(fwd_hook)
h2 = target_layer.register_full_backward_hook(bwd_hook)

logits = model(x)
cls = logits.argmax(1).item()
score = logits[0, cls]
score.backward()

A = acts[0].detach().cpu().numpy()[0]
G = grads[0].detach().cpu().numpy()[0]
weights = G.mean(axis=(1, 2))
cam = np.maximum((weights[:, None, None] * A).sum(axis=0), 0)
cam = cv2.resize(cam, (img0.shape[1], img0.shape[0]))
cam = (cam - cam.min()) / (cam.max() + 1e-6)

heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
overlay = (0.4 * heatmap + 0.6 * img0).astype(np.uint8)
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(args.output, overlay)

h1.remove()
h2.remove()
print(f"Saved CAM -> {args.output}")
