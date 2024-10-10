from pathlib import Path
from typing import Any

import numpy as np
import torch
from pytorch_lightning.core.module import LightningModule
from timm.utils import ModelEmaV3
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from torchmetrics import Accuracy

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
        self.val_acc = Accuracy(task="multiclass", num_classes=6)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        augment_policy = get_augment_policy(self.cfg)

        if augment_policy == "mixup":
            x, y = mixup(x, y)
        elif augment_policy == "nothing":
            pass
        else:
            raise ValueError(f"unknown augment policy {augment_policy}")

        output = self.model(x)
        loss_dict = {k: v if k == "loss" else v.detach() for k, v in self.loss(output, y).items()}
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

        loss_dict = self.loss(output, y)
        log_dict = {"val_" + k: v for k, v in loss_dict.items()}
        self.log_dict(log_dict, on_epoch=True, sync_dist=True)

        output = torch.argmax(output, dim=-1)
        self.val_acc(output, y)
        self.log("val_acc", self.val_acc, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self):
        pass

    def on_test_start(self):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def on_test_epoch_end(self):
        pass

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        x, idx, instance_numbers = batch

        # x: b=1, depth, h, w
        if self.cfg.test.tta:
            outputs = []
            output = self.model(x)  # b=1, depth, 6
            outputs.append(output)
            output = self.model(x.flip(3))
            outputs.append(output)
            output = self.model(x.flip(1)).flip(1)
            outputs.append(output)
            output = self.model(x.flip(1).flip(3)).flip(1)
            outputs.append(output)
            output = torch.mean(torch.stack(outputs), dim=0)
        else:
            output = self.model(x)  # b=1, depth, 6

        output = torch.softmax(output, dim=-1).cpu().numpy()
        return output, idx, instance_numbers

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
