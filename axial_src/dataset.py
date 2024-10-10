from pathlib import Path
import numpy as np
import cv2
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class MyDataset(Dataset):
    def __init__(self, cfg, ids, mode="train", img_dir=None):
        assert mode in ["train", "val", "test"]
        self.cfg = cfg
        self.ids = ids
        self.mode = mode
        self.transforms = get_train_transforms(cfg) if mode == "train" else get_val_transforms(cfg)

        if img_dir is not None:
            self.img_dir = img_dir
        else:
            self.img_dir = Path(__file__).parents[1].joinpath("input", cfg.task.dirname)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        filename = self.ids[idx]
        img_path = self.img_dir.joinpath(filename)
        d = np.load(img_path)
        img = d["voxel"]
        y = d["class_ids"] if self.mode in ["train", "val"] else -100
        depth = self.cfg.task.depth

        if self.mode == "train":
            if np.random.randint(2) == 1:
                img = np.flip(img, axis=2).copy()
                y = np.flip(y, axis=0).copy()

            if img.shape[2] < depth:
                z = depth - img.shape[2]
                img = np.pad(img, ((0, 0), (0, 0), (0, z)), mode="constant", constant_values=0)
                y = np.pad(y, (0, z), mode="constant", constant_values=-100)
            else:
                z = np.random.randint(0, img.shape[2] - depth + 1)
                img = img[:, :, z:z + depth]
                y = y[z:z + depth]

        img = self.transforms(image=img)["image"]

        if self.mode in ["train", "val"]:
            return img, y
        else:
            return img, filename, d["instance_numbers"]


def get_train_transforms(cfg):
    return A.Compose(
        [
            A.Resize(height=cfg.task.img_size, width=cfg.task.img_size, p=1),
            # A.CenterCrop(height=cfg.task.img_size, width=cfg.task.img_size, p=1),
            A.ShiftScaleRotate(p=0.5, border_mode=cv2.BORDER_CONSTANT, shift_limit=0.1, scale_limit=0.2, value=0,
                               rotate_limit=30, mask_value=0),
            # A.RandomScale(scale_limit=(0.8, 1.2), p=1),
            # A.PadIfNeeded(min_height=cfg.task.img_size, min_width=cfg.task.img_size, p=1.0,
            #               border_mode=cv2.BORDER_CONSTANT, value=0),
            # A.RandomCrop(height=self.cfg.data.train_img_h, width=self.cfg.data.train_img_w, p=1.0),
            # A.MultiplicativeNoise(multiplier=(0.9, 1.1), elementwise=True, p=0.5),
            # A.RandomRotate90(p=1.0),
            A.HorizontalFlip(p=0.5 if cfg.task.dirname.startswith("axial") else 0),
            # A.VerticalFlip(p=0.5),
            # A.RandomBrightnessContrast(p=0.5, brightness_limit=0.1, contrast_limit=0.1),
            # A.HueSaturationValue(p=0.5),
            # A.ToGray(p=0.3),
            # A.GaussNoise(var_limit=(0.0, 0.05), p=0.5),
            # A.GaussianBlur(p=0.5),
            # normalize with imagenet statis
            A.Normalize(p=1.0, mean=0.5, std=0.25, max_pixel_value=1.0),
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
            A.Normalize(p=1.0, mean=0.5, std=0.25, max_pixel_value=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )
