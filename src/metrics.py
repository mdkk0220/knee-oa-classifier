import torch
import torch.nn.functional as F

@torch.no_grad()
def accuracy(pred_logits_or_labels: torch.Tensor, target: torch.Tensor) -> float:
    p = pred_logits_or_labels
    t = target
    if p.ndim == 2:  # (N,C) ·ÎÁþÀÌ¸é argmax
        p = p.argmax(1)
    return float((p == t).float().mean().item())

def classification_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(pred_logits, target)
