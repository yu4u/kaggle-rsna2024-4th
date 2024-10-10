from pathlib import Path
import re
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedGroupKFold
from torch.utils.data import DataLoader
import torch
from pytorch_lightning import LightningDataModule

from .dataset import MyDataset


class MyDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.root_dir = Path(__file__).parents[1].joinpath("input", cfg.task.dirname)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit":
            df = pd.read_csv(self.root_dir.joinpath("metadata.csv"))
            # ["study_id", "series_id", "filename", "level", "left", "right"]
            train = df[df["fold_id"] != self.cfg.data.fold_id]
            val = df[df["fold_id"] == self.cfg.data.fold_id]

            if self.cfg.data.train_all:
                train_ids = df["filename"].values
            else:
                train_ids = train["filename"].values

            val_ids = val["filename"].values
            train_targets = train[["left_subarticular_stenosis", "right_subarticular_stenosis"]].values
            val_targets = val[["left_subarticular_stenosis", "right_subarticular_stenosis"]].values
            self.train_dataset = MyDataset(self.cfg, train_ids, train_targets, "train")
            self.val_dataset = MyDataset(self.cfg, val_ids, val_targets, "val")
            print(f"train: {len(train_ids)}, val: {len(val_ids)}")
        elif stage in ["test", "predict"]:
            target = self.cfg.test.target
            dirname = f"{target}_keypoint_{self.cfg.test.mode}_dataset"
            csv_path = Path(__file__).parents[1].joinpath(dirname, "metadata.csv")
            df = pd.read_csv(csv_path)

            if self.cfg.test.mode == "val":
                df = df[df["fold_id"] == self.cfg.data.fold_id]

            test_ids = df["filename"].values
            self.test_dataset = MyDataset(self.cfg, test_ids, None,"test", csv_path.parent)
            print(f"test: {len(test_ids)}")
        else:
            raise ValueError(f"unknown stage {stage}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.data.batch_size, collate_fn=None,
                          shuffle=True, drop_last=True, num_workers=self.cfg.data.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.data.batch_size, collate_fn=None,
                          shuffle=False, drop_last=False, num_workers=self.cfg.data.num_workers)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.data.batch_size, collate_fn=None,
                          shuffle=False, drop_last=False, num_workers=self.cfg.data.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.cfg.data.batch_size, collate_fn=None,
                          shuffle=False, drop_last=False, num_workers=self.cfg.data.num_workers)
