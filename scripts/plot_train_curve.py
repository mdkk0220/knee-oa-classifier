# scripts/plot_train_curve.py
import re
import matplotlib.pyplot as plt
from pathlib import Path

# 로그 파일 경로
log_path = Path("outputs/resnet50_finetune_combined/train.log")
out_path = Path("outputs/vis/week6/train_curve.png")
out_path.parent.mkdir(parents=True, exist_ok=True)

# 로그 파일 읽기
with open(log_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# 패턴 예시:
# [Epoch 3/20] Train Loss: 0.4567 Acc: 82.15% | Val Loss: 0.4891 Acc: 80.24%
pattern = re.compile(
    r"Epoch\s+(\d+)/\d+\]\s+Train Loss:\s+([\d.]+).*Acc:\s+([\d.]+)%\s+\|\s+Val Loss:\s+([\d.]+).*Acc:\s+([\d.]+)%"
)

epochs, train_loss, val_loss, train_acc, val_acc = [], [], [], [], []

for line in lines:
    match = pattern.search(line)
    if match:
        ep, tl, ta, vl, va = match.groups()
        epochs.append(int(ep))
        train_loss.append(float(tl))
        train_acc.append(float(ta))
        val_loss.append(float(vl))
        val_acc.append(float(va))

# 그래프 그리기
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label="Train Loss", marker="o")
plt.plot(epochs, val_loss, label="Val Loss", marker="o")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, label="Train Acc", marker="o")
plt.plot(epochs, val_acc, label="Val Acc", marker="o")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.tight_layout()
plt.savefig(out_path, dpi=150)
print(f"✅ 학습 곡선 저장 완료 → {out_path}")
