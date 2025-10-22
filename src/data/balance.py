import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def make_sampler(labels):
    class_sample_count = np.array([(labels == c).sum() for c in np.unique(labels)], dtype=float)
    class_weights = 1.0 / (class_sample_count + 1e-6)
    sample_weights = np.array([class_weights[lbl] for lbl in labels])
    sample_weights = torch.from_numpy(sample_weights).double()
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
