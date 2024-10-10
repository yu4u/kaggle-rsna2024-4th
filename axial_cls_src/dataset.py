from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class MyDataset(Dataset):
    def __init__(self, cfg, ids, targets=None, mode="train", img_dir=None):
        assert mode in ["train", "val", "test"]
        self.cfg = cfg
        self.ids = ids
        self.targets = targets
        self.mode = mode
        self.transforms = get_train_transforms(cfg) if mode == "train" else get_val_transforms(cfg)

        if img_dir is None:
            self.img_dir = Path(__file__).parents[1].joinpath("input", cfg.task.dirname)
        else:
            self.img_dir = img_dir

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_filename = self.ids[idx]
        img_path = self.img_dir.joinpath(img_filename)

        if img_path.suffix == ".npz":
            if self.mode == "train":
                if np.random.randint(2) == 1:
                    img = np.load(str(img_path))["img"]
                else:
                    try:
                        img = np.load(str(img_path))["gt_img"]
                    except KeyError:
                        img = np.load(str(img_path))["img"]
            else:
                img = np.load(str(img_path))["img"]
        else:
            img = cv2.imread(str(img_path))

        if self.mode == "test":
            sample = self.transforms(image=img)
            img = sample["image"]
            return img, img_filename

        def str_to_class_id(s):
            if s == "Normal/Mild":
                return 0
            elif s == "Moderate":
                return 1
            elif s == "Severe":
                return 2
            elif s == "nan":
                return -100
            else:
                raise ValueError(f"Invalid class: {s}")

        y = self.targets[idx]
        y = torch.tensor([str_to_class_id(str(s)) for s in y])

        if self.mode == "train":
            if np.random.randint(2) == 1:
                img = np.flip(img, axis=2).copy()

            if self.cfg.task.dirname.startswith("axial"):
                if self.cfg.task.train_target != "axial_crop":
                    if np.random.randint(2) == 1:
                        img = np.flip(img, axis=1).copy()
                        y = y.flip(0)

        img = self.transforms(image=img)["image"]
        return img, y


def get_train_transforms(cfg):
    return A.Compose(
        [
            A.Resize(height=cfg.task.img_size, width=cfg.task.img_size, p=1),
            # A.CenterCrop(height=cfg.task.img_size, width=cfg.task.img_size, p=1),
            A.ShiftScaleRotate(p=0.5, border_mode=cv2.BORDER_CONSTANT, shift_limit=0.0, scale_limit=0.1, value=0,
                               rotate_limit=30, mask_value=0),
            # A.RandomScale(scale_limit=(0.8, 1.2), p=1),
            # A.PadIfNeeded(min_height=cfg.task.img_size, min_width=cfg.task.img_size, p=1.0,
            #               border_mode=cv2.BORDER_CONSTANT, value=0),
            # A.RandomCrop(height=self.cfg.data.train_img_h, width=self.cfg.data.train_img_w, p=1.0),
            # A.MultiplicativeNoise(multiplier=(0.9, 1.1), elementwise=True, p=0.5),
            # A.RandomRotate90(p=1.0),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            # A.RandomBrightnessContrast(p=0.5, brightness_limit=0.1, contrast_limit=0.1),
            # A.HueSaturationValue(p=0.5),
            # A.ToGray(p=0.3),
            # A.GaussNoise(var_limit=(0.0, 0.05), p=0.5),
            # A.GaussianBlur(p=0.5),
            # normalize with imagenet statis
            A.Normalize(p=1.0, mean=0.5, std=0.25),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )


def get_val_transforms(cfg):
    return A.Compose(
        [
            A.Resize(height=cfg.task.img_size, width=cfg.task.img_size, p=1),
            # A.RandomScale(scale_limit=(1.0, 1.0), p=1),
            # A.PadIfNeeded(min_height=cfg.task.img_size, min_width=cfg.task.img_size, p=1.0,
            #              border_mode=cv2.BORDER_CONSTANT, value=0),
            # A.Crop(y_max=self.cfg.data.val_img_h, x_max=self.cfg.data.val_img_w, p=1.0),
            A.Normalize(p=1.0, mean=0.5, std=0.25),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )
