import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.data.balance import make_sampler
from src.data.dataset import XrayDataset
from src.data.transforms import get_train_aug, get_val_aug
from src.models.resnet50 import ResNet50KL
from src.train.metrics import macro_f1, qwk


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    tr_ds = XrayDataset(
        cfg["data"]["manifest"],
        cfg["data"]["img_root"],
        "train",
        get_train_aug(cfg["data"]["input_size"]),
    )
    va_ds = XrayDataset(
        cfg["data"]["manifest"],
        cfg["data"]["img_root"],
        "val",
        get_val_aug(cfg["data"]["input_size"]),
    )

    if cfg["train"].get("class_weight", False):
        sampler = make_sampler(torch.tensor(tr_ds.labels))
        tr_loader = DataLoader(
            tr_ds,
            batch_size=cfg["data"]["batch_size"],
            sampler=sampler,
            num_workers=cfg["data"]["num_workers"],
        )
    else:
        tr_loader = DataLoader(
            tr_ds,
            batch_size=cfg["data"]["batch_size"],
            shuffle=True,
            num_workers=cfg["data"]["num_workers"],
        )
    va_loader = DataLoader(
        va_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50KL(cfg["model"]["num_classes"], cfg["model"]["dropout"]).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["amp"])

    best_qwk, best_path = -1.0, out_dir / "best.ckpt"

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for imgs, labels, _ in tr_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg["train"]["amp"]):
                logits = model(imgs)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, labels, _ in va_loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                preds = logits.argmax(1).cpu().tolist()
                y_pred += preds
                y_true += labels.tolist()
        q = qwk(y_true, y_pred)
        f1 = macro_f1(y_true, y_pred)
        print(f"[Epoch {epoch+1}] QWK={q:.4f}  F1={f1:.4f}")

        if q > best_qwk:
            best_qwk = q
            torch.save(model.state_dict(), best_path)
            print(f"Saved: {best_path} (QWK {best_qwk:.4f})")


if __name__ == "__main__":
    main()
