import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import KFold, GroupKFold
from tqdm import tqdm

from axial_src.util import get_img_from_dcm, normalize_img
from axial_cls_src.util import target_names


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dirname", type=str, default="axial_cls_all_dataset")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--mode", type=str, default="val")  # val or test
    parser.add_argument("--channel_num", type=int, default=3)
    args = parser.parse_args()
    return args


def compute_similarity_transform(x1, y1, x2, y2, w, h):
    target_a = (0.25 * w, 0.5 * h)
    target_b = (0.75 * w, 0.5 * h)

    dist_original = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    dist_target = np.sqrt((target_b[0] - target_a[0]) ** 2 + (target_b[1] - target_a[1]) ** 2)

    scale = dist_target / dist_original

    angle_original = np.arctan2(y2 - y1, x2 - x1)
    angle_target = np.arctan2(target_b[1] - target_a[1], target_b[0] - target_a[0])
    rotation_angle = angle_target - angle_original
    translation_x = target_a[0] - (x1 * scale * np.cos(rotation_angle) - y1 * scale * np.sin(rotation_angle))
    translation_y = target_a[1] - (x1 * scale * np.sin(rotation_angle) + y1 * scale * np.cos(rotation_angle))

    transform_matrix = np.array([
        [scale * np.cos(rotation_angle), -scale * np.sin(rotation_angle), translation_x],
        [scale * np.sin(rotation_angle), scale * np.cos(rotation_angle), translation_y]
    ])

    return transform_matrix


def apply_similarity_transform(image, transform_matrix, w, h):
    transformed_image = cv2.warpAffine(image, transform_matrix, (w, h), flags=cv2.INTER_LINEAR)
    return transformed_image


def main():
    args = get_args()
    img_size = args.img_size
    mode = args.mode  # val or test
    channel_num = args.channel_num
    root_dir = Path(__file__).parent.joinpath("input")
    coord_df = pd.read_csv(root_dir.joinpath("train_label_coordinates.csv"))
    coord_df = coord_df[coord_df["condition"].str.contains("Subarticular Stenosis")]
    output_dir = root_dir.joinpath(args.dirname)
    output_dir.mkdir(exist_ok=True)

    for series_id, sub_df in tqdm(coord_df.groupby("series_id")):
        study_id = sub_df["study_id"].values[0]
        reverse = False

        if len(sub_df) < 10:
            continue

        sub_df = sub_df.sort_values("instance_number")

        if sub_df["level"].values.tolist() != ['L1/L2', 'L1/L2', 'L2/L3', 'L2/L3', 'L3/L4', 'L3/L4', 'L4/L5', 'L4/L5',
                                               'L5/S1', 'L5/S1']:
            sub_df = sub_df.sort_values("instance_number", ascending=False)

            if sub_df["level"].values.tolist() != ['L1/L2', 'L1/L2', 'L2/L3', 'L2/L3', 'L3/L4', 'L3/L4', 'L4/L5',
                                                   'L4/L5', 'L5/S1', 'L5/S1']:
                continue
            else:
                reverse = True

        instance_numbers = sub_df["instance_number"].values.reshape(-1, 2)
        diff = np.diff(instance_numbers, axis=1)

        if np.abs(diff).max() > 2:
            continue

        instance_numbers = instance_numbers.mean(axis=1).astype(int)
        center_x = sub_df["x"].values.reshape(-1, 2).astype(int)
        center_y = sub_df["y"].values.reshape(-1, 2).astype(int)

        for li, (instance_number, x, y) in enumerate(zip(instance_numbers, center_x, center_y)):
            prefix = "train" if mode == "val" else "test"
            dcm_paths = sorted(Path(f"input/{prefix}_images/{study_id}/{series_id}").glob("*.dcm"),
                               key=lambda x: int(x.stem))

            try:
                idx = [int(p.stem) for p in dcm_paths].index(instance_number)
            except ValueError:
                try:
                    idx = [int(p.stem) for p in dcm_paths].index(instance_number - 1)
                except ValueError:
                    print(f"ValueError: {instance_number}")
                    continue

            imgs = []

            # get n images
            for i in range(channel_num):
                try:
                    dcm_path = dcm_paths[idx - channel_num // 2 + i]
                except IndexError:
                    dcm_path = dcm_paths[idx]
                    print(f"IndexError: {idx - channel_num // 2 + i}")

                img = get_img_from_dcm(dcm_path, normalize=True, to_uint8=False)
                imgs.append(img)

            for i in range(channel_num):
                if imgs[i].shape != imgs[channel_num // 2].shape:
                    imgs[i] = imgs[channel_num // 2].copy()

            img = np.stack(imgs, -1)

            if x[0] > x[1]:
                x = x[::-1]
                y = y[::-1]

            transform_matrix = compute_similarity_transform(x[0], y[0], x[1], y[1], img_size, img_size)
            transformed_image = apply_similarity_transform(img, transform_matrix, img_size, img_size)
            output_img_paths = sorted(output_dir.glob(f"{study_id}_{series_id}_{li}_*.npz"))

            if len(output_img_paths) == 0:
                print(f"{study_id}_{series_id}_{li} does not exist")
                continue

            output_img_path = output_img_paths[0]
            transformed_image = normalize_img(transformed_image)
            transformed_image = transformed_image * 255
            img = np.load(output_img_path)["img"]
            np.savez(output_img_path, img=img, gt_img=transformed_image)


if __name__ == '__main__':
    main()
