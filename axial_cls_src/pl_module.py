from pathlib import Path
from typing import Any
import sklearn.metrics
import numpy as np
import torch
from pytorch_lightning.core.module import LightningModule
from timm.utils import ModelEmaV3
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

from .model import get_model_from_cfg, EnsembleModel
from .loss import get_loss
from .util import mixup, get_augment_policy


class MyModel(LightningModule):
    def __init__(self, cfg, mode="train"):
        super().__init__()
        self.preds = None
        self.gts = None
        self.cfg = cfg

        if mode == "test":
            self.model = EnsembleModel(cfg)
        else:
            self.model = get_model_from_cfg(cfg, cfg.model.resume_path)

        if mode != "test" and cfg.model.ema:
            self.model_ema = ModelEmaV3(
                self.model,
                decay=cfg.model.ema_decay,
                update_after_step=cfg.model.ema_update_after_step,
            )

        self.loss = get_loss(cfg)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        augment_policy = get_augment_policy(self.cfg)

        if augment_policy == "mixup":
            x, targets1, targets2, lam = mixup(x, y)
        elif augment_policy == "nothing":
            pass
        else:
            raise ValueError(f"unknown augment policy {augment_policy}")

        output = self.model(x)
        output = output.reshape(-1, 3)

        if augment_policy == "nothing":
            y = y.reshape(-1)
            loss_dict = {k: v if k == "loss" else v.detach() for k, v in self.loss(output, y).items()}
        else:
            targets1 = targets1.reshape(-1)
            targets2 = targets2.reshape(-1)
            loss_dict1 = self.loss(output, targets1)
            loss_dict2 = self.loss(output, targets2)
            loss_dict = {k: lam * loss_dict1[k] + (1 - lam) * loss_dict2[k] for k in loss_dict1.keys()}
            loss_dict = {k: v if k == "loss" else v.detach() for k, v in loss_dict.items()}

        self.log_dict(loss_dict, on_epoch=True, sync_dist=True)
        return loss_dict

    def on_train_batch_end(self, out, batch, batch_idx):
        if self.cfg.model.ema:
            self.model_ema.update(self.model)

    def on_validation_epoch_start(self) -> None:
        self.preds = []
        self.gts = []

    def validation_step(self, batch, batch_idx):
        x, y = batch

        if self.cfg.model.ema:
            output = self.model_ema.module(x)
        else:
            output = self.model(x)

        output = output.reshape(-1, 3)
        y = y.reshape(-1)
        loss_dict = self.loss(output, y)
        log_dict = {"val_" + k: v for k, v in loss_dict.items()}
        self.log_dict(log_dict, on_epoch=True, sync_dist=True)

        output = torch.softmax(output, dim=-1)
        self.preds.append(output.detach().cpu().float().numpy())
        self.gts.append(y.detach().cpu().float().numpy())

    def on_validation_epoch_end(self):
        preds = np.concatenate(self.preds)
        gts = np.concatenate(self.gts)
        gt_to_weight = {0: 1.0, 1: 2.0, 2: 4.0, -100: 0}
        sample_weights = [gt_to_weight[gt] for gt in gts]

        try:
            score = sklearn.metrics.log_loss(gts, preds, sample_weight=sample_weights,
                                             labels=range(3))
        except ValueError as e:
            print(e)
            score = 1.0

        self.log("val_score", score, on_epoch=True, sync_dist=True)

    def on_test_start(self):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def on_test_epoch_end(self):
        pass

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        x, filenames = batch

        # x: b, c, h, w
        if self.cfg.test.tta:
            outputs = []

            output = self.model(x)
            outputs.append(output)

            output = self.model(x.flip(1))
            outputs.append(output)

            if self.cfg.test.target == "axial":
                output = self.model(x.flip(-1))
                output = output.reshape(-1, 2, 3)
                output = output.flip(1).reshape(-1, 6)
                outputs.append(output)


                output = self.model(x.flip(1).flip(-1))
                output = output.reshape(-1, 2, 3)
                output = output.flip(1).reshape(-1, 6)
                outputs.append(output)
            
            output = torch.mean(torch.stack(outputs), dim=0)
        else:
            output = self.model(x)  # b, 6

        # output = output.reshape(-1, 2, 3)
        # output = torch.softmax(output, dim=-1)
        results = []

        for filename, pred in zip(filenames, output.cpu().numpy()):
            # filename = f"{study_id}_{series_id}_{part_id}_{instance_number}.npz"
            parts = filename.replace(".npz", "").split("_")

            if len(parts) == 4:
                study_id, series_id, part_id, instance_number = parts
                results.append([int(study_id), int(series_id), int(part_id), int(instance_number), *pred])
            elif len(parts) == 5:
                study_id, series_id, part_id, instance_number, side = parts
                results.append([int(study_id), int(series_id), int(part_id), int(instance_number), side, *pred])
            else:
                raise ValueError(f"unknown filename {filename}")

        return results

    def configure_optimizers(self):
        optimizer = create_optimizer_v2(model_or_params=self.model, **self.cfg.opt)
        batch_size = self.cfg.data.batch_size
        updates_per_epoch = len(self.trainer.datamodule.train_dataset) // batch_size // self.trainer.num_devices
        scheduler, num_epochs = create_scheduler_v2(optimizer=optimizer, num_epochs=self.cfg.trainer.max_epochs,
                                                    warmup_lr=0, **self.cfg.scheduler,
                                                    step_on_epochs=False, updates_per_epoch=updates_per_epoch)
        lr_dict = dict(
            scheduler=scheduler,
            interval="step",
            frequency=1,  # same as default
        )
        return dict(optimizer=optimizer, lr_scheduler=lr_dict)

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step_update(num_updates=self.global_step)
