import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_aug(size=512):
    return A.Compose(
        [
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.Resize(size, size),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=5, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_val_aug(size=512):
    return A.Compose(
        [
            A.Resize(size, size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
