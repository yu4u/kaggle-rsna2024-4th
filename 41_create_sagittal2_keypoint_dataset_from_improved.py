import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from axial_src.util import get_img_from_dcm, normalize_img
from axial_cls_src.util import target_names


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dirname", type=str, default="sagittal2_keypoint_dataset")
    parser.add_argument("--img_size", type=int, default=512)
    args = parser.parse_args()
    return args


def create_gaussian_patch(patch_size, sigma):
    """
    Create a Gaussian patch with the specified size and standard deviation.

    :param patch_size: Size of the square patch (patch_size, patch_size).
    :param sigma: Standard deviation of the Gaussian.
    :return: Gaussian patch as a 2D numpy array.
    """
    center = patch_size // 2
    x = np.arange(patch_size) - center
    y = np.arange(patch_size) - center
    x, y = np.meshgrid(x, y)
    gaussian = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    gaussian *= 255.0 / np.max(gaussian)
    return gaussian.astype(np.uint8)


def place_patch(image, patch, x, y):
    """
    Place a Gaussian patch on the image at the specified position.

    :param image: The base image where the patch will be placed.
    :param patch: The Gaussian patch to be placed.
    :param x: X-coordinate where the patch will be centered.
    :param y: Y-coordinate where the patch will be centered.
    """
    patch_size = patch.shape[0]
    half_size = patch_size // 2

    # Determine the region of the image to place the patch
    x_start = max(0, x - half_size)
    x_end = min(image.shape[1], x + half_size)
    y_start = max(0, y - half_size)
    y_end = min(image.shape[0], y + half_size)

    # Calculate the corresponding region in the patch
    patch_x_start = half_size - (x - x_start)
    patch_x_end = half_size + (x_end - x)
    patch_y_start = half_size - (y - y_start)
    patch_y_end = half_size + (y_end - y)

    # Place the patch onto the image
    image[y_start:y_end, x_start:x_end] = patch[patch_y_start:patch_y_end, patch_x_start:patch_x_end]


def main():
    args = get_args()
    img_size = args.img_size
    root_dir = Path(__file__).parent.joinpath("input")
    coord_df = pd.read_csv(Path(__file__).parent.joinpath("misc", "coords_rsna_improved.csv"))
    coord_df = coord_df[coord_df["condition"].str.contains("Spinal Canal Stenosis")]
    output_dir = root_dir.joinpath(args.dirname)
    output_dir.mkdir(exist_ok=True)
    gaussian_patch = create_gaussian_patch(45, 9)
    rows = []

    for series_id, sub_df in tqdm(coord_df.groupby("series_id")):
        sub_df = sub_df[sub_df["side"] == "R"]
        study_id = sub_df["study_id"].values[0]

        if len(sub_df) < 5:
            print(f"skipped: {series_id}")
            continue

        sub_df = sub_df.sort_values("level")

        instance_number = int(sub_df["instance_number"].values.mean())

        if instance_number == 1:
            print(f"invalid instance number, skipped: {series_id}")
            continue

        dcm_path = Path(f"input/train_images/{study_id}/{series_id}/{instance_number}.dcm")

        if not dcm_path.exists():
            print(f"base file not found: {dcm_path}")
            continue

        imgs = []

        dcm_paths = sorted(Path(f"input/train_images/{study_id}/{series_id}").glob("*.dcm"), key=lambda x: int(x.stem))
        # get index of current instance_number
        idx = [int(p.stem) for p in dcm_paths].index(instance_number)

        # get 3 images
        for i in range(3):
            try:
                dcm_path = dcm_paths[idx - 1 + i]
            except IndexError:
                dcm_path = dcm_paths[idx]
                print(f"IndexError: {idx - 1 + i}")

            img = get_img_from_dcm(dcm_path, normalize=True, to_uint8=False)
            imgs.append(img)

        for i in range(3):
            if imgs[i].shape != imgs[1].shape:
                imgs[i] = imgs[1].copy()

        img = np.stack(imgs, -1).astype(np.float32)
        h, w = img.shape[:2]
        img = cv2.resize(img, (img_size, img_size))
        # img = normalize_img(img)
        img_filename = f"{study_id}_{series_id}_{instance_number}_img.npy"
        output_img_path = output_dir.joinpath(img_filename)
        np.save(output_img_path, img)

        masks = []

        for _, row in sub_df.iterrows():
            x = row["relative_x"] * img_size
            y = row["relative_y"] * img_size
            mask = np.zeros((img_size, img_size), np.uint8)
            place_patch(mask, gaussian_patch, int(x), int(y))
            masks.append(mask)

        mask_filename = f"{study_id}_{series_id}_{instance_number}_mask.npy"
        output_mask_path = output_dir.joinpath(mask_filename)
        np.save(output_mask_path, np.stack(masks, 0))

        target = ["Normal/Mild"] * 5  # dummy
        rows.append([study_id, series_id, img_filename, *target])

    df = pd.DataFrame(rows, columns=["study_id", "series_id", "filename"] + target_names)
    split = pd.read_csv("misc/train_with_split.csv")
    df = df.merge(split[["study_id", "fold_id"]], on="study_id", how="left")
    df.to_csv(output_dir.joinpath("metadata.csv"), index=False)


if __name__ == '__main__':
    main()
