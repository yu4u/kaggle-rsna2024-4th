import torch.nn as nn


def get_loss(cfg):
    return MyLoss(cfg)


class MyLoss(nn.Module):
    def __init__(self, cfg):
        super(MyLoss, self).__init__()
        self.cfg = cfg
        self.loss = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        return_dict = dict()
        loss = self.loss(y_pred.transpose(1, 2), y_true)
        return_dict["loss"] = loss
        return return_dict


def main():
    pass


if __name__ == '__main__':
    main()
