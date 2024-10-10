import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

from axial_src.util import get_img_from_dcm, normalize_img
from axial_cls_src.util import target_names


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dirname", type=str, default="axial_cls_all_dataset")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--keypoint_img_size", type=int, default=512)
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

    if mode == "val":
        dfs = [pd.read_csv(f"axial_val_keypoint_preds_fold{i}.csv") for i in range(5)]
        df = pd.concat(dfs, axis=0)
        output_dir = root_dir.joinpath(args.dirname)
    else:
        df = pd.read_csv("axial_test_keypoint_preds.csv")
        output_dir = Path(f"axial_cls_test_dataset")

    train_df = pd.read_csv("misc/train_with_split.csv")
    output_dir.mkdir(exist_ok=True)
    rows = []

    for row in tqdm(df.itertuples(index=False), total=len(df)):
        study_id, series_id, instance_number, part_id, left_x, left_y, right_x, right_y = row[:8]
        prefix = "train" if mode == "val" else "test"
        dcm_paths = sorted(Path(f"input/{prefix}_images/{study_id}/{series_id}").glob("*.dcm"),
                           key=lambda x: int(x.stem))
        idx = [int(p.stem) for p in dcm_paths].index(instance_number)
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
        h, w = img.shape[:2]
        left_x = left_x * w / args.keypoint_img_size
        left_y = left_y * h / args.keypoint_img_size
        right_x = right_x * w / args.keypoint_img_size
        right_y = right_y * h / args.keypoint_img_size

        transform_matrix = compute_similarity_transform(left_x, left_y, right_x, right_y, img_size, img_size)
        transformed_image = apply_similarity_transform(img, transform_matrix, img_size, img_size)
        li_to_level = {0: "l1_l2", 1: "l2_l3", 2: "l3_l4", 3: "l4_l5", 4: "l5_s1"}
        level = li_to_level[part_id]

        if mode == "val":
            target_column_names = [f"{prefix}_{level}" for prefix in target_names]
            target = train_df[train_df["study_id"] == study_id][target_column_names + ["fold_id"]].values[0]
        else:
            target = [0] * (len(target_names) + 1)

        filename = f"{study_id}_{series_id}_{part_id}_{instance_number}.npz"
        output_img_path = output_dir.joinpath(filename)
        transformed_image = normalize_img(transformed_image)
        transformed_image = transformed_image * 255
        np.savez(output_img_path, img=transformed_image)
        rows.append([study_id, series_id, filename, level, *target])

    df = pd.DataFrame(rows, columns=["study_id", "series_id", "filename", "level"] + target_names + ["fold_id"])
    df.to_csv(output_dir.joinpath("metadata.csv"), index=False)


if __name__ == '__main__':
    main()
