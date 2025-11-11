# src/data/dataset.py 파일 수정 (XrayDataset 클래스)

from pathlib import Path
import cv2
import pandas as pd # (필요없지만 다른 함수와 호환을 위해 남겨둠)
from torch.utils.data import Dataset, DataLoader
from src.data.transforms import get_train_aug, get_val_aug
import os # 폴더를 탐색하기 위해 추가

class XrayDataset(Dataset):
    """
    Knee OA X-ray Dataset
    - 폴더 구조 기반 데이터 로더 (예: img_root/split/label/image.png)
    - 흑백 이미지를 RGB 변환하여 ResNet 입력과 호환
    """
    # manifest_csv를 제거하고 img_root와 split만 사용합니다.
    # __init__ 정의가 get_dataloaders와 호환되도록 매개변수는 유지했습니다.
    def __init__(self, manifest_csv, img_root, split="train", transform=None):
        
        self.img_root = Path(img_root) # 예: '../data/images' 폴더
        self.split = split
        self.transform = transform
        
        self.paths = []
        self.labels = []
        
        # 🚨 여기서 데이터를 직접 수집합니다! 🚨
        split_dir = self.img_root / split # 예: '../data/images/train'
        
        # KL 등급(0부터 4까지) 폴더를 탐색합니다.
        for label_int in range(5):
            label_name = str(label_int) # '0', '1', '2', '3', '4'
            label_dir = split_dir / label_name # 예: '../data/images/train/0'
            
            # 해당 폴더에 있는 모든 .png 파일을 찾습니다.
            if label_dir.is_dir():
                for img_path in label_dir.glob('*.png'):
                    self.paths.append(img_path)
                    self.labels.append(label_int) # 폴더 이름(0~4)을 레이블로 사용

        # 만약 데이터가 없으면 오류를 출력합니다.
        if not self.paths:
             raise FileNotFoundError(f"경로 {split_dir}에서 이미지를 찾을 수 없습니다. 경로와 폴더 구조를 확인하세요.")


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # ... (나머지 __getitem__ 코드는 그대로 사용 가능) ...
        path = str(self.paths[idx])
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        return img, label


# get_dataloaders 함수는 그대로 유지해도 됩니다.
def get_dataloaders(manifest, img_root, input_size, batch_size, num_workers=4):
    """
    이제 manifest CSV 파일이 없어도 폴더 구조를 기반으로 작동합니다.
    """
    train_t = get_train_aug(input_size)
    val_t = get_val_aug(input_size)

    # manifest 매개변수는 더 이상 사용하지 않지만, 함수 호출 호환을 위해 유지합니다.
    train_ds = XrayDataset(manifest, img_root, split="train", transform=train_t)
    val_ds = XrayDataset(manifest, img_root, split="val", transform=val_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
