# src/train/train.py
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.models.resnet50 import ResNet50KL
from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.data.dataset import get_dataloaders


def train_model(config_path: str = "src/config/train_resnet50.yaml"):
    # 1️⃣ Config 로드
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ⚙️ 안전한 float 변환 (문자열 방지)
    config["train"]["lr"] = float(config["train"]["lr"])
    config["train"]["weight_decay"] = float(config["train"]["weight_decay"])

    # 2️⃣ Seed 고정
    seed = config["seed"]
    set_seed(seed)

    # 3️⃣ 디바이스 및 로거 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger()
    logger.info("🔧 Device: %s", device)
    logger.info("⚙️ Config loaded from: %s", config_path)

    # 4️⃣ 데이터 로더 준비
    train_loader, val_loader = get_dataloaders(
        manifest=config["data"]["manifest"],
        img_root=config["data"]["img_root"],
        input_size=config["data"]["input_size"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )
    logger.info("📦 Dataloader ready (Train=%d, Val=%d)", len(train_loader.dataset), len(val_loader.dataset))

    # 5️⃣ 모델 생성
    model_cfg = config["model"]
    model = ResNet50KL(
        num_classes=model_cfg["num_classes"],
        dropout=model_cfg["dropout"],
        freeze_backbone=False,
    ).to(device)
    logger.info("🧠 Model initialized: %s", model_cfg["name"])

    # 6️⃣ 손실함수
    criterion = nn.CrossEntropyLoss()
    if config["train"]["class_weight"]:
        logger.info("📊 Class weighting enabled (placeholder: implement after label analysis)")

    # 7️⃣ Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )
    logger.info(
        "🚀 Optimizer: AdamW | lr=%.6f, weight_decay=%.6f",
        config["train"]["lr"],
        config["train"]["weight_decay"],
    )

    if config["train"]["scheduler"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["train"]["epochs"]
        )
        logger.info("📈 Scheduler: CosineAnnealingLR")
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        logger.info("📈 Scheduler: StepLR")

    # 8️⃣ Mixed Precision (AMP)
    scaler = torch.cuda.amp.GradScaler(enabled=config["train"]["amp"])
    logger.info("⚡ AMP enabled: %s", config["train"]["amp"])

    # 9️⃣ 학습 루프
    best_metric = -1.0
    epochs = config["train"]["epochs"]
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model_best.pth")

    logger.info("🏁 Training started for %d epochs", epochs)

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=config["train"]["amp"]):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        train_acc = correct / total * 100
        train_loss = running_loss / len(train_loader)

        # 🔹 검증
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = outputs.max(1)
                val_total += labels.size(0)
                val_correct += preds.eq(labels).sum().item()

        val_acc = val_correct / val_total * 100
        val_loss /= len(val_loader)
        scheduler.step()

        logger.info(
            f"[Epoch {epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%"
        )

        # 🔹 최고 성능 모델 저장
        if val_acc > best_metric:
            best_metric = val_acc
            torch.save(model.state_dict(), model_path)
            logger.info(f"💾 Best model saved ({val_acc:.2f}%)")

    logger.info(f"🎯 Training completed. Best Val Acc: {best_metric:.2f}%")


if __name__ == "__main__":
    train_model()
