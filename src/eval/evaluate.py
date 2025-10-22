import argparse

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import XrayDataset
from src.data.transforms import get_val_aug
from src.models.resnet50 import ResNet50KL
from src.train.metrics import macro_f1, qwk

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
args = ap.parse_args()
cfg = yaml.safe_load(open(args.config))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds = XrayDataset(
    cfg["data"]["manifest"], cfg["data"]["img_root"], "val", get_val_aug(cfg["data"]["input_size"])
)
loader = DataLoader(
    ds, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"]
)

model = ResNet50KL(cfg["model"]["num_classes"], cfg["model"]["dropout"]).to(device)
model.load_state_dict(torch.load(f"{cfg['output_dir']}/best.ckpt", map_location=device))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for x, y, _ in loader:
        x = x.to(device)
        pred = model(x).argmax(1).cpu().tolist()
        y_pred += pred
        y_true += y.tolist()

print("QWK:", qwk(y_true, y_pred))
print("Macro-F1:", macro_f1(y_true, y_pred))
