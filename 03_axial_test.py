import argparse
import numpy as np
from pytorch_lightning import Trainer
from omegaconf import OmegaConf

from axial_src.datamodule import MyDataModule
from axial_src.pl_module import MyModel


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, default="axial_src/config.yaml")
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
    all_preds = trainer.predict(model, datamodule=dm)
    results = dict()
    filenames = []

    for preds, indices, instance_number_list in all_preds:
        for pred, idx, instance_numbers in zip(preds, indices, instance_number_list):
            results[idx] = pred
            results[f"{idx}_instance_numbers"] = instance_numbers
            filenames.append(idx)

    mode = cfg.test.mode

    if mode == "val":
        fold_id = cfg.data.fold_id
        np.savez(f"{target}_{mode}_preds_fold{fold_id}.npz", **results, filenames=filenames)
    else:
        np.savez(f"{target}_{mode}_preds.npz", **results, filenames=filenames)


if __name__ == '__main__':
    main()
