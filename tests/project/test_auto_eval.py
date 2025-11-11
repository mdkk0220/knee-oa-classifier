from src.eval.auto_eval import compare_models
from torch.utils.data import DataLoader, TensorDataset
import torch

def test_compare_models():
    class Dummy(torch.nn.Module):
        def forward(self, x): return torch.randn(len(x), 3)

    models = [Dummy(), Dummy()]
    data = DataLoader(TensorDataset(torch.randn(10, 4), torch.randint(0,3,(10,))))
    result = compare_models(models, data)
    assert isinstance(result, dict)
