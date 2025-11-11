from typing import Dict, List
import torch
from torch.utils.data import DataLoader
from src.metrics import classification_loss

def train_one_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: str='cpu') -> Dict[str, float]:
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = classification_loss(logits, y)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().item()) * x.size(0)
        n += x.size(0)
    return {'loss': total / max(1, n)}

def dummy_train_loop(epochs: int=3) -> List[float]:
    # 아주 단순한 더미 학습 로그 (테스트용)
    return [1.0/(e+1) for e in range(max(0, int(epochs)))]
