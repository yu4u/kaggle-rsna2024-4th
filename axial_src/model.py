from pathlib import Path
import math
import numpy as np
import torch
import torch.nn as nn
import timm
from torch.nn import Transformer
import torch.nn.functional as F
# from mmaction.models import ResNet3dCSN


class SlidingSlicer(nn.Module):
    def __init__(self, slice_size=3):
        super(SlidingSlicer, self).__init__()

        # Create convolution layer to simulate the sliding slice operation
        self.conv = nn.Conv3d(1, slice_size, kernel_size=(slice_size, 1, 1), stride=1, bias=False,
                              padding=(slice_size // 2, 0, 0))

        # Set weights to simulate identity operation and bias to 0
        with torch.no_grad():
            self.conv.weight.data.fill_(0)
            for i in range(slice_size):
                self.conv.weight.data[i, 0, i] = 1

        for param in self.conv.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.conv(x)
        out = out.transpose(1, 2)
        return out


class AxialModel(nn.Module):
    def __init__(self, cfg, pretrained):
        super().__init__()
        self.cfg = cfg
        in_channels = cfg.task.in_channels

        if in_channels != 1:
            self.slicer = SlidingSlicer(slice_size=in_channels)

        model = timm.create_model(cfg.model.backbone, pretrained=pretrained, num_classes=0, in_chans=in_channels)
        self.model = model
        hidden_dim = model.num_features

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=cfg.task.layer_num,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(inplace=True),
            nn.Dropout(cfg.model.drop_rate),
            nn.Linear(hidden_dim, 6))

    def forward(self, x):
        b, depth, h, w = x.shape

        if self.cfg.task.in_channels == 1:
            x = x.reshape(b * depth, 1, h, w)
        else:
            x = x.unsqueeze(1)
            x = self.slicer(x)
            x = x.reshape(b * depth, self.cfg.task.in_channels, h, w)

        x = self.model(x)  # (b * c, hidden_dim)
        x = x.reshape(b, depth, -1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


class AxialModel2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = ResNet3dCSN(
            depth=50,
            pretrained="https://download.openmmlab.com/mmaction/recognition/csn/ircsn_from_scratch_r50_ig65m_20210617-ce545a37.pth",
            pretrained2d=False,
            temporal_strides=(1, 1, 1, 1),
            bottleneck_mode="ir",
            # norm_eval=True,
            # bn_frozen=True,
            in_channels=1,
            conv1_stride_t=1,
            pool1_stride_t=1,
            with_pool2=False,
        )
        self.model.init_weights()
        hidden_dim = self.model.inplanes

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(inplace=True),
            nn.Dropout(cfg.model.drop_rate),
            nn.Linear(hidden_dim, 6))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.model(x)  # ([b, 2048, 32, 2, 2])
        x = x.mean([3, 4])
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        return x


def get_model_from_cfg(cfg, resume_path=None):
    pretrained = True if resume_path is None else False

    if cfg.model.arch == "2d":
        model = AxialModel(cfg, pretrained)
    elif cfg.model.arch == "3d":
        model = AxialModel2(cfg)
    else:
        raise ValueError(f"unknown arch {cfg.model.arch}")

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
        outputs = [model(x) for model in self.models]
        x = torch.mean(torch.stack(outputs), dim=0)
        return x
