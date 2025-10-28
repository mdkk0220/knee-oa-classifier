# src/utils/scheduler.py
from torch.optim.lr_scheduler import CosineAnnealingLR

def get_scheduler(optimizer, config):
    """학습률 스케줄러 생성"""
    if config.get("scheduler", "cosine") == "cosine":
        return CosineAnnealingLR(
            optimizer, T_max=config.get("epochs", 20), eta_min=1e-6
        )
    return None
