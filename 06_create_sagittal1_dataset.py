import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

from axial_src.util import get_img_from_dcm, normalize_img


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dirname", type=str, default="sagittal1_dataset")
    parser.add_argument("--img_size", type=int, default=128)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    img_size = args.img_size
    root_dir = Path(__file__).parent.joinpath("input")
    coord_df = pd.read_csv(root_dir.joinpath("train_label_coordinates.csv"))
    coord_df = coord_df[coord_df["condition"].str.contains("Neural Foraminal")]
    output_dir = root_dir.joinpath(args.dirname)
    output_dir.mkdir(exist_ok=True)
    rows = []

    for series_id, sub_df in tqdm(coord_df.groupby("series_id")):
        study_id = sub_df["study_id"].values[0]

        if len(sub_df) < 10:
            continue

        left_df = sub_df[sub_df["condition"].str.contains("Left")]
        right_df = sub_df[sub_df["condition"].str.contains("Right")]
        left_in = int(left_df["instance_number"].values.mean())
        right_in = int(right_df["instance_number"].values.mean())

        dcm_paths = sorted(root_dir.joinpath(f"train_images/{study_id}/{series_id}").glob("*.dcm"),
                           key=lambda x: int(x.stem))

        voxel = []

        for dcm_path in dcm_paths:
            img = get_img_from_dcm(dcm_path, normalize=True, to_uint8=False)
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            voxel.append(img)

        class_ids = np.zeros(len(voxel), dtype=int)

        if left_in < right_in:
            class_ids[left_in:right_in] = 1
        else:
            class_ids[right_in:left_in] = 1

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
