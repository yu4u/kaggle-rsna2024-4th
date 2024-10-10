import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.losses import DiceLoss, BINARY_MODE


def get_loss(cfg):
    def my_loss(y_pred, y_true, weight):
        mask_pred, label_pred = y_pred
        mask, label = y_true
        # dice = DiceLoss(BINARY_MODE, from_logits=True)(y_pred, y_true)
        # loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")
        label_pred = label_pred.reshape(-1, 3, 2)
        ce_loss = F.cross_entropy(label_pred, label, weight=torch.tensor([1.0, 2.0, 4.0], device=label.device))

        if cfg.task.dirname.startswith("axial"):
            bce_loss = F.mse_loss(mask_pred, mask, reduction="none")
            bce_loss = bce_loss * weight[:, :, None, None]
            bce_loss = bce_loss.mean()
        else:
            bce_loss = F.mse_loss(mask_pred, mask)

        loss = ce_loss + bce_loss * 100
        return dict(loss=loss, ce_loss=ce_loss, bce_loss=bce_loss)
    return my_loss


class MyLoss(nn.Module):
    def __init__(self, cfg):
        super(MyLoss, self).__init__()
        self.cfg = cfg
        self.loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 4.0]))

    def forward(self, y_pred, y_true):
        return_dict = dict()
        loss = self.loss(y_pred, y_true)
        return_dict["loss"] = loss
        return return_dict


def main():
    pass


if __name__ == '__main__':
    main()
