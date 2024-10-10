import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import KFold
from tqdm import tqdm

from axial_src.util import get_img_from_dcm, normalize_img


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dirname", type=str, default="axial_dataset")
    parser.add_argument("--img_size", type=int, default=128)
    args = parser.parse_args()
    return args


def get_class_ids(dcm_paths, thresholds, reverse=False):
    class_ids = []

    if reverse:
        op = lambda x, y: x < y
    else:
        op = lambda x, y: x >= y

    current_group = 0

    for dcm_path in dcm_paths:
        num = int(dcm_path.stem)

        while current_group < len(thresholds) and op(num, thresholds[current_group]):
            current_group += 1

        class_ids.append(current_group)

    return class_ids


def main():
    args = get_args()
    img_size = args.img_size
    root_dir = Path(__file__).parent.joinpath("input")
    coord_df = pd.read_csv(root_dir.joinpath("train_label_coordinates.csv"))
    coord_df = coord_df[coord_df["condition"].str.contains("Subarticular Stenosis")]
    output_dir = root_dir.joinpath(args.dirname)
    output_dir.mkdir(exist_ok=True)
    rows = []

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

        instance_numbers = instance_numbers.mean(-1)
        dcm_paths = sorted(root_dir.joinpath(f"train_images/{study_id}/{series_id}").glob("*.dcm"),
                           key=lambda x: int(x.stem), reverse=reverse)

        class_ids = get_class_ids(dcm_paths, instance_numbers, reverse)
        voxel = []

        for class_id, dcm_path in zip(class_ids, dcm_paths):
            img = get_img_from_dcm(dcm_path, normalize=True, to_uint8=False)
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            voxel.append(img)

        npz_filename = f"{study_id}_{series_id}.npz"
        rows.append([study_id, series_id, npz_filename])
        voxel = np.stack(voxel, -1)
        # voxel = normalize_img(voxel)
        npz_path = output_dir.joinpath(npz_filename)
        np.savez(npz_path, voxel=voxel, class_ids=class_ids)

    df = pd.DataFrame(rows, columns=["study_id", "series_id", "filename"])
    split = pd.read_csv("misc/train_with_split.csv")
    df = df.merge(split[["study_id", "fold_id"]], on="study_id", how="left")
    df.to_csv(output_dir.joinpath("metadata.csv"), index=False)


if __name__ == '__main__':
    main()
