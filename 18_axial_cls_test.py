import argparse
from pathlib import Path
from itertools import chain
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from omegaconf import OmegaConf

from axial_cls_src.datamodule import MyDataModule
from axial_cls_src.pl_module import MyModel


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, default="axial_cls_src/config.yaml")
    parser.add_argument("--sep", action="store_true")
    parser.add_argument("--output_suffix", type=str, default="baseline")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    print(OmegaConf.to_yaml(cfg))
    trainer = Trainer(**cfg.trainer)
    dm = MyDataModule(cfg)
    model = MyModel(cfg, mode="test")
    results = trainer.predict(model, datamodule=dm)
    results = list(chain.from_iterable(results))
    target = cfg.test.target
    mode = cfg.test.mode

    if len(results[0]) == 10:
        columns =["study_id", "series_id", "part_id", "instance_number",
                  "left0", "left1", "left2", "right0", "right1", "right2"]
    elif len(results[0]) == 11:
        columns = ["study_id", "series_id", "part_id", "instance_number", "side",
                   "left0", "left1", "left2", "right0", "right1", "right2"]
    else:
        raise ValueError(f"Invalid results: {results[0]}")

    df = pd.DataFrame(data=results, columns=columns)

    if mode == "val":
        fold_id = cfg.data.fold_id
        df.to_csv(f"{target}_{mode}_cls_preds_{args.output_suffix}_fold{fold_id}.csv", index=False)
    else:
        df.to_csv(f"{target}_{mode}_cls_preds_{args.output_suffix}.csv", index=False)


if __name__ == '__main__':
    main()
