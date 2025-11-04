# src/data/dataset.py
from pathlib import Path
import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from src.data.transforms import get_train_aug, get_val_aug


class XrayDataset(Dataset):
    """
    Knee OA X-ray Dataset
    - manifest CSV 기반 데이터 로더
    - split 컬럼으로 train/val 구분
    - 흑백 이미지를 RGB 변환하여 ResNet 입력과 호환
    - Albumentations 기반 transform 사용
    """

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
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # X-ray → 단일 채널
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)   # ResNet 호환 3채널 변환
        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        return img, label


def get_dataloaders(manifest, img_root, input_size, batch_size, num_workers=4):
    """
    manifest CSV 기반 train/val DataLoader 구성
    - manifest CSV에 split 컬럼이 있어야 함 ("train", "val")
    """
    train_t = get_train_aug(input_size)
    val_t = get_val_aug(input_size)

    train_ds = XrayDataset(manifest, img_root, split="train", transform=train_t)
    val_ds = XrayDataset(manifest, img_root, split="val", transform=val_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
