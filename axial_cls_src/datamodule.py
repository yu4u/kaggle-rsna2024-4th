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
            train_ids = train["filename"].values
            val_ids = val["filename"].values

            if self.cfg.task.dirname.startswith("axial"):
                if self.cfg.task.train_target == "axial":
                    target_columns = ["left_subarticular_stenosis", "right_subarticular_stenosis"]
                elif self.cfg.task.train_target == "axial_crop":
                    target_columns = ["target", "target"]
                elif self.cfg.task.train_target == "sagittal1":
                    target_columns = ["left_neural_foraminal_narrowing", "right_neural_foraminal_narrowing"]
                elif self.cfg.task.train_target == "sagittal2":
                    target_columns = ["spinal_canal_stenosis", "spinal_canal_stenosis"]
                else:
                    raise ValueError(f"unknown train_target {self.cfg.task.train_target}")
            else:
                target_columns = ["target", "target"]

            train_targets = train[target_columns].values
            val_targets = val[target_columns].values
            self.train_dataset = MyDataset(self.cfg, train_ids, train_targets, "train")
            self.val_dataset = MyDataset(self.cfg, val_ids, val_targets, "val")
            print(f"train: {len(train_ids)}, val: {len(val_ids)}")
        elif stage in ["test", "predict"]:
            mode = self.cfg.test.mode
            target = self.cfg.test.target

            if mode == "val":
                root_dir = self.root_dir
                df = pd.read_csv(root_dir.joinpath("metadata.csv"))
                df = df[df["fold_id"] == self.cfg.data.fold_id]
            else:
                if self.cfg.test.dirname is not None:
                    dirname = self.cfg.test.dirname
                else:
                    dirname = f"{target}_cls_test_dataset"

                root_dir = Path(__file__).parents[1].joinpath(dirname)
                df = pd.read_csv(root_dir.joinpath("metadata.csv"))
                print(f"loaded {root_dir.joinpath('metadata.csv')}")

            test_ids = df["filename"].values
            self.test_dataset = MyDataset(self.cfg, test_ids, None, "test", root_dir)
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
        return DataLoader(self.test_dataset, batch_size=self.cfg.data.batch_size, collate_fn=None,
                          shuffle=False, drop_last=False, num_workers=self.cfg.data.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.cfg.data.batch_size, collate_fn=None,
                          shuffle=False, drop_last=False, num_workers=self.cfg.data.num_workers)
