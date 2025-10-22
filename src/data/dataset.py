from pathlib import Path

import cv2
import pandas as pd
from torch.utils.data import Dataset


class XrayDataset(Dataset):
    def __init__(self, manifest_csv, img_root, split="train", transform=None):
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.img_root = Path(img_root)
        self.transform = transform

        self.labels = self.df["label"].astype(int).tolist()
        self.paths = [self.img_root / fp for fp in self.df["filepath"].tolist()]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = str(self.paths[idx])
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # 1채널
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # ResNet 호환 3채널
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(image=img)
            img = sample["image"]
        return img, label, path
