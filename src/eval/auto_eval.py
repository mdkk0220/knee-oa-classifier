from typing import Dict, List
import torch
from torch.utils.data import DataLoader
from src.metrics import classification_loss

@torch.no_grad()
def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: str='cpu') -> Dict[str, float]:
    model.eval()
    n, correct, total_loss = 0, 0.0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += float(classification_loss(logits, y).item()) * x.size(0)
        correct += float((logits.argmax(1) == y).float().sum().item())
        n += x.size(0)
    return {'loss': total_loss/max(1,n), 'acc': correct/max(1,n), 'n': n}

def compare_models(models: List[torch.nn.Module], loader: DataLoader, device: str='cpu') -> Dict[str, float]:
    best = None
    for i, m in enumerate(models):
        r = evaluate_model(m, loader, device=device)
        r['model_index'] = i
        if best is None or (r['acc'], -r['loss']) > (best['acc'], -best['loss']):
            best = r
    return best or {'acc':0.0,'loss':1e9,'n':0,'model_index':-1}
