import argparse
from pathlib import Path
from itertools import chain
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from omegaconf import OmegaConf

from axial_keypoint_src.datamodule import MyDataModule
from axial_keypoint_src.pl_module import MyModel


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, default="axial_keypoint_src/config.yaml")
    parser.add_argument("--sep", action="store_true")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    print(OmegaConf.to_yaml(cfg))
    target = cfg.test.target
    trainer = Trainer(**cfg.trainer)
    dm = MyDataModule(cfg)
    model = MyModel(cfg, mode="test")
    results = trainer.predict(model, datamodule=dm)
    results = list(chain.from_iterable(results))

    if target == "axial":
        columns =["study_id", "series_id", "instance_number", "part_id", "left_x", "left_y", "right_x", "right_y",
                  "left0", "left1", "left2", "right0", "right1", "right2"]
    elif target == "sagittal1":
        columns = ["study_id", "series_id", "instance_number", "side", "part_id", "x", "y"]
    elif target == "sagittal2":
        columns = ["study_id", "series_id", "instance_number", "part_id", "x", "y"]
    else:
        raise ValueError(f"unknown target {target}")

    df = pd.DataFrame(data=results, columns=columns)
    mode = cfg.test.mode

    if mode == "val":
        fold_id = cfg.data.fold_id
        df.to_csv(f"{target}_{mode}_keypoint_preds_fold{fold_id}.csv", index=False)
    else:
        df.to_csv(f"{target}_{mode}_keypoint_preds.csv", index=False)


if __name__ == '__main__':
    main()
