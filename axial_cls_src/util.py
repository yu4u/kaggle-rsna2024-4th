import numpy as np
import pydicom
import torch


target_names = [
    "spinal_canal_stenosis",
    "left_neural_foraminal_narrowing",
    "right_neural_foraminal_narrowing",
    "left_subarticular_stenosis",
    "right_subarticular_stenosis"
]


def get_first_of_dicom_field_as_int(x):
    if isinstance(x, pydicom.multival.MultiValue):
        return int(x[0])
    else:
        return int(x)


def get_img_from_dcm(img_path):
    dcm = pydicom.dcmread(img_path)
    pixel_array = dcm.pixel_array

    intercept = float(dcm.RescaleIntercept) if hasattr(dcm, "RescaleIntercept") else 0
    slope = float(dcm.RescaleSlope) if hasattr(dcm, "RescaleSlope") else 1
    center = int(dcm.WindowCenter)
    width = int(dcm.WindowWidth)
    low = center - width / 2
    high = center + width / 2
    pixel_array = (pixel_array * slope) + intercept
    pixel_array = np.clip(pixel_array, low, high)

    """
    # use 1 and 99 percentile as low and high
    low = np.percentile(pixel_array, 1)
    high = np.percentile(pixel_array, 99)
    pixel_array = np.clip(pixel_array, 0, 255)
    """

    pixel_array = (pixel_array - low) / (high - low) * 255.0
    pixel_array = pixel_array.astype(np.uint8)

    return pixel_array


def mixup(data, targets, alpha=1.0):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    lam = np.random.beta(alpha, alpha)
    data = data * lam + shuffled_data * (1 - lam)
    return data, targets, shuffled_targets, lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(data, targets, alpha=1.0):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    return data, targets, shuffled_targets, lam


def get_augment_policy(cfg):
    p_mixup = cfg.loss.mixup
    p_cutmix = cfg.loss.cutmix
    p_nothing = 1 - p_mixup - p_cutmix
    return np.random.choice(["nothing", "mixup", "cutmix"], p=[p_nothing, p_mixup, p_cutmix], size=1)[0]


def main():
    pass


if __name__ == '__main__':
    main()