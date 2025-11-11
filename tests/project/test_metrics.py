import torch
from src.metrics import accuracy

def test_accuracy_basic():
    y_pred = torch.tensor([0, 1, 2, 2])
    y_true = torch.tensor([0, 1, 1, 2])
    assert abs(accuracy(y_pred, y_true) - 0.75) < 1e-5
