from pathlib import Path
import math
import numpy as np
import torch
import torch.nn as nn
import timm
import segmentation_models_pytorch as smp


def get_model_from_cfg(cfg, resume_path=None):
    encoder_weights = "imagenet" if resume_path is None else None
    classes = 2 if cfg.task.dirname.startswith("axial") else 5

    if cfg.test.target != "axial":
        classes = 5

    model = getattr(smp, cfg.model.arch)(cfg.model.backbone, classes=classes, in_channels=3,
                                         encoder_weights=encoder_weights,
                                         aux_params=dict(pooling="avg", classes=6))

    if resume_path:
        print(f"loading model from {str(resume_path)}")
        checkpoint = torch.load(str(resume_path), map_location="cpu")

        if np.any([k.startswith("model_ema.") for k in checkpoint["state_dict"].keys()]):
            state_dict = {k[17:]: v for k, v in checkpoint["state_dict"].items() if k.startswith("model_ema.")}
        else:
            state_dict = {k[6:]: v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}

        model.load_state_dict(state_dict, strict=True)

    return model


class EnsembleModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if Path(cfg.model.resume_path).is_dir():
            resume_paths = Path(cfg.model.resume_path).rglob("*.ckpt")
        else:
            resume_paths = [Path(cfg.model.resume_path)]

        self.models = nn.ModuleList()

        for resume_path in resume_paths:
            model = get_model_from_cfg(cfg, resume_path)
            self.models.append(model)

    def __call__(self, x):
        seg_preds, cls_preds = [], []

        for model in self.models:
            seg_pred, cls_pred = model(x)
            seg_preds.append(seg_pred)
            cls_preds.append(cls_pred)

        seg_preds = torch.mean(torch.stack(seg_preds), dim=0)
        cls_preds = torch.mean(torch.stack(cls_preds), dim=0)

        return seg_preds, cls_preds
