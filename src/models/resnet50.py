# src/models/resnet50.py
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50KL(nn.Module):
    """
    ResNet50 기반 Transfer Learning 모델
    - Grad-CAM 호환
    - Fine-tuning 및 Feature Extraction 지원
    """
    def __init__(self, num_classes: int = 5, dropout: float = 0.3, freeze_backbone: bool = False):
        super().__init__()
        # 1. Pretrained ResNet50 로드
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_feats = self.backbone.fc.in_features

        # 2. Backbone 일부 동결
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 3. Custom Classification Head
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_feats, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
