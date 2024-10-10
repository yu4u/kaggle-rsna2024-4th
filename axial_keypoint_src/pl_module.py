from pathlib import Path
from typing import Any
import sklearn.metrics
import numpy as np
import cv2
import torch
import torch.nn.functional as F
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
        self.k_preds = None
        self.k_gts = None
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
        x, y, cls_y, weight = batch
        augment_policy = get_augment_policy(self.cfg)

        if augment_policy == "mixup":
            x, targets1, targets2, lam = mixup(x, y)
        elif augment_policy == "nothing":
            pass
        else:
            raise ValueError(f"unknown augment policy {augment_policy}")

        output = self.model(x)
        y = y, cls_y

        if augment_policy == "nothing":
            loss_dict = {k: v if k == "loss" else v.detach() for k, v in self.loss(output, y, weight).items()}
        else:
            loss_dict1 = self.loss(output, targets1, weight)
            loss_dict2 = self.loss(output, targets2, weight)
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
        self.k_preds = []
        self.k_gts = []

    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum((a - b) ** 2, axis=1))

    def validation_step(self, batch, batch_idx):
        x, y, cls_y, weight = batch

        if self.cfg.model.ema:
            output = self.model_ema.module(x)
        else:
            output = self.model(x)

        loss_dict = self.loss(output, (y, cls_y), weight)
        log_dict = {"val_" + k: v for k, v in loss_dict.items()}
        self.log_dict(log_dict, on_epoch=True, sync_dist=True)

        output_dir = Path(__file__).parents[1].joinpath("input", self.cfg.test.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        output1 = output[0]
        output = output[1]

        output = output.reshape(-1, 3, 2)
        output = output.transpose(2, 1)  # (N, 2, 3)
        output = torch.softmax(output, dim=-1)
        self.preds.append(output.detach().cpu().float().numpy())
        self.gts.append(cls_y.detach().cpu().float().numpy())

        if self.cfg.task.dirname.startswith("axial"):
            for i, (gt_i, pred_i, w_i) in enumerate(zip(y.float().cpu().numpy(), output1.float().cpu().numpy(), weight.cpu().numpy())):
                # keypoint
                min_val, max_val, min_loc, (pred_x, pred_y) = cv2.minMaxLoc(pred_i[w_i[1]])
                min_val, max_val, min_loc, (gt_x, gt_y) = cv2.minMaxLoc(gt_i[w_i[1]])
                self.k_gts.append([gt_x, gt_y])
                self.k_preds.append([pred_x, pred_y])
        else:
            for i, (gt_i, pred_i) in enumerate(zip(y.float().cpu().numpy(), output1.float().cpu().numpy())):
                for gt_l, pred_l in zip(gt_i, pred_i):
                    min_val, max_val, min_loc, (pred_x, pred_y) = cv2.minMaxLoc(pred_l)
                    min_val, max_val, min_loc, (gt_x, gt_y) = cv2.minMaxLoc(gt_l)
                    self.k_gts.append([gt_x, gt_y])
                    self.k_preds.append([pred_x, pred_y])

                gt_i = np.concatenate(gt_i, axis=0)
                pred_i = np.concatenate(pred_i, axis=0)
                img = np.concatenate([gt_i, pred_i], axis=1)
                img = np.clip(img, 0, 1)
                img = (img * 255).astype(np.uint8)
                img_path = output_dir.joinpath(f"{batch_idx}_{i}.png")
                cv2.imwrite(str(img_path), img)

    @staticmethod
    def get_score(gts, preds):
        if len(gts.shape) == 2:
            gts = gts.reshape(-1)
            preds = preds.reshape(-1, 3)

        gt_to_weight = {0: 1.0, 1: 2.0, 2: 4.0, -100: 0}
        sample_weights = [gt_to_weight[gt] for gt in gts]

        try:
            score = sklearn.metrics.log_loss(gts, preds, sample_weight=sample_weights,
                                             labels=range(3))
        except ValueError as e:
            print(e)
            score = 1.0

        return score

    def on_validation_epoch_end(self):
        preds = np.concatenate(self.preds)
        gts = np.concatenate(self.gts)
        score = self.get_score(gts, preds)
        self.log("val_score", score, on_epoch=True, sync_dist=True)

        preds = np.array(self.k_preds)
        gts = np.array(self.k_gts)
        score = self.euclidean_distance(preds, gts).mean()
        self.log("val_keypoint_error", score, on_epoch=True, sync_dist=True)

    def on_test_start(self):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def on_test_epoch_end(self):
        pass

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        x, filenames = batch
        target = self.cfg.test.target

        # x: b, depth, h, w
        if self.cfg.test.tta:
            outputs = []
            output = self.model(x)  # (b, 2, h, w), (b, 6)
            outputs.append(output[0])
            output = self.model(x.flip(3))
            outputs.append(output[0].flip(1).flip(3))
            output = self.model(x.flip(1))
            outputs.append(output[0])
            output = self.model(x.flip(1).flip(3))
            outputs.append(output[0].flip(1).flip(3))
            seg_preds = torch.mean(torch.stack(outputs), dim=0)
        else:
            output = self.model(x)
            seg_preds = output[0]

        cls_preds = output[1]
        cls_preds = cls_preds.reshape(-1, 3, 2)
        cls_preds = cls_preds.transpose(2, 1)
        results = []

        if target == "axial":
            for filename, pred, cls_pred in zip(filenames, seg_preds.cpu().numpy(), cls_preds.cpu().numpy()):
                min_val, max_val, min_loc, (left_x, left_y) = cv2.minMaxLoc(pred[0])
                min_val, max_val, min_loc, (right_x, right_y) = cv2.minMaxLoc(pred[1])
                # img_filename = f"{study_id}_{series_id}_{instance_number}_{part_id}_img.png"

                study_id, series_id, instance_number, part_id, _ = filename.split("_")
                results.append(
                    [int(study_id), int(series_id), int(instance_number), int(part_id), left_x, left_y, right_x, right_y,
                     *cls_pred.flatten()])
        elif target == "sagittal1":
            for filename, pred, cls_pred in zip(filenames, seg_preds.cpu().numpy(), cls_preds.cpu().numpy()):
                study_id, series_id, instance_number, side, _ = filename.split("_")

                for part_id, pred_l in enumerate(pred):
                    min_val, max_val, min_loc, (x, y) = cv2.minMaxLoc(pred_l)
                    results.append([int(study_id), int(series_id), int(instance_number), side, int(part_id), x, y])
        elif target == "sagittal2":
            for filename, pred, cls_pred in zip(filenames, seg_preds.cpu().numpy(), cls_preds.cpu().numpy()):
                # img_filename = f"{study_id}_{series_id}_{instance_number}_img.npy"
                study_id, series_id, instance_number, _ = filename.split("_")

                for part_id, pred_l in enumerate(pred):
                    min_val, max_val, min_loc, (x, y) = cv2.minMaxLoc(pred_l)
                    results.append([int(study_id), int(series_id), int(instance_number), int(part_id), x, y])
        else:
            raise ValueError(f"unknown target {target}")

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
