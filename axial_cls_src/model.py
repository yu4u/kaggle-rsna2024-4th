from pathlib import Path
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention_weights = nn.Linear(input_dim, 1)  # 重みを計算するための線形層

    def forward(self, x):
        attention_scores = self.attention_weights(x)
        attention_weights = F.softmax(attention_scores, dim=1)
        weighted_sum = torch.sum(attention_weights * x, dim=1)
        return weighted_sum


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

    def forward(self, x):
        with torch.no_grad():
            x = x.unsqueeze(1)
            out = self.conv(x)
            out = out.transpose(1, 2)
        return out


class AxialClsModel(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.cfg = cfg
        in_channels = cfg.model.in_channels
        self.in_channels = in_channels
        self.stem = SlidingSlicer(in_channels)

        if self.is_transformer():
            model = timm.create_model(cfg.model.backbone, pretrained=pretrained, num_classes=0, in_chans=in_channels,
                                      img_size=cfg.task.img_size)
        else:
            model = timm.create_model(cfg.model.backbone, pretrained=pretrained, num_classes=0, in_chans=in_channels)

        self.model = model
        hidden_dim = model.num_features
        self.attention_pooling = AttentionPooling(hidden_dim)

        self.fc = nn.Sequential(
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            # nn.ELU(inplace=True),
            nn.Dropout(cfg.model.drop_rate),
            nn.Linear(hidden_dim, 6))

    def is_transformer(self):
        substrings = ["swin", "vit"]
        return any(substring in self.cfg.model.backbone for substring in substrings)

    def forward(self, x):
        b, depth, h, w = x.shape
        x = self.stem(x)
        x = x.reshape(b * depth, self.in_channels, h, w)
        x = self.model(x)  # (b * depth, hidden_dim)
        x = x.reshape(b, depth, -1)
        x = self.attention_pooling(x)
        x = self.fc(x)
        return x


class AxialClsModel2(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.cfg = cfg
        in_channels = cfg.model.in_channels
        self.in_channels = in_channels
        self.stem = SlidingSlicer(in_channels)

        if self.is_transformer():
            model = timm.create_model(cfg.model.backbone, pretrained=pretrained, num_classes=0, in_chans=in_channels,
                                      img_size=cfg.task.img_size)
        else:
            model = timm.create_model(cfg.model.backbone, pretrained=pretrained, num_classes=0, in_chans=in_channels)

        self.model = model
        hidden_dim = model.num_features
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention_pooling = AttentionPooling(hidden_dim * 2)

        self.fc = nn.Sequential(
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            # nn.ELU(inplace=True),
            nn.Dropout(cfg.model.drop_rate),
            nn.Linear(hidden_dim * 2, 6))

    def is_transformer(self):
        substrings = ["swin", "vit"]
        return any(substring in self.cfg.model.backbone for substring in substrings)

    def forward(self, x):
        b, depth, h, w = x.shape
        x = self.stem(x)
        x = x.reshape(b * depth, self.in_channels, h, w)
        x = self.model(x)  # (b * depth, hidden_dim)
        x = x.reshape(b, depth, -1)
        x, _ = self.lstm(x)
        x = self.attention_pooling(x)
        x = self.fc(x)
        return x


class AxialClsModel3(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from mmaction.models import ResNet3dCSN
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
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            # nn.ELU(inplace=True),
            nn.Dropout(cfg.model.drop_rate),
            nn.Linear(hidden_dim, 6))

        self.attention_pooling = AttentionPooling(hidden_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.model(x)  # ([b, 2048, 32, 2, 2])
        x = x.flatten(start_dim=2, end_dim=4)
        x = x.permute(0, 2, 1)
        x = self.attention_pooling(x)
        x = self.fc(x)
        return x


def get_model_from_cfg(cfg, resume_path=None):
    pretrained = True if resume_path is None else False

    if cfg.model.arch == "2d":
        model = timm.create_model(cfg.model.backbone, pretrained=pretrained, num_classes=6, in_chans=3)
    elif cfg.model.arch == "2.5d":
        model = AxialClsModel(cfg, pretrained)
    elif cfg.model.arch == "3d":
        model = AxialClsModel3(cfg)
    elif cfg.model.arch == "lstm":
        model = AxialClsModel2(cfg, pretrained)
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
