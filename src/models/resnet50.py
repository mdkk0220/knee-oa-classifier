import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50KL(nn.Module):
    def __init__(self, num_classes: int = 5, dropout: float = 0.2):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)
