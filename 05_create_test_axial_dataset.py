import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from joblib import Parallel, delayed

from axial_src.util import get_img_from_dcm


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--mode", type=str, default="val")  # val or test
    parser.add_argument("--target", type=str, default="axial")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    img_size = args.img_size
    mode = args.mode  # val or test
    target = args.target  # axial, sagittal1, sagittal2
    root_dir = Path(__file__).parent.joinpath("input")
    prefix = "train" if mode == "val" else "test"  # train or test
    series_df = pd.read_csv(root_dir.joinpath(f"{prefix}_series_descriptions.csv"))
    target_to_description = {
        "axial": "Axial T2",
        "sagittal1": "Sagittal T1",
        "sagittal2": "Sagittal T2/STIR",
    }
    series_df = series_df[series_df["series_description"] == target_to_description[target]]
    output_dir = Path(f"{target}_{mode}_dataset")
    output_dir.mkdir(exist_ok=True)

    def save_npz(row):
        series_id = row["series_id"]
        study_id = row["study_id"]
        dcm_paths = sorted(root_dir.joinpath(f"{prefix}_images/{study_id}/{series_id}").glob("*.dcm"),
                           key=lambda x: int(x.stem))
        voxel = []
        instance_numbers = [int(dcm_path.stem) for dcm_path in dcm_paths]

        for dcm_path in dcm_paths:
            img = get_img_from_dcm(dcm_path, normalize=True, to_uint8=False)
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            voxel.append(img)

        npz_filename = f"{study_id}_{series_id}.npz"
        voxel = np.stack(voxel, -1)
        # voxel = normalize_img(voxel)
        npz_path = output_dir.joinpath(npz_filename)
        np.savez(npz_path, voxel=voxel, instance_numbers=instance_numbers)
        return [study_id, series_id, npz_filename]

    rows = Parallel(n_jobs=-1)([delayed(save_npz)(row) for _, row in tqdm(series_df.iterrows(), total=len(series_df))])

    """
    rows = []
        
    for _, row in tqdm(series_df.iterrows(), total=len(series_df)):
        series_id = row["series_id"]
        study_id = row["study_id"]
        dcm_paths = sorted(root_dir.joinpath(f"{prefix}_images/{study_id}/{series_id}").glob("*.dcm"),
                           key=lambda x: int(x.stem))
        voxel = []

        for dcm_path in dcm_paths:
            img = get_img_from_dcm(dcm_path, normalize=True, to_uint8=False)
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            voxel.append(img)

        npz_filename = f"{study_id}_{series_id}.npz"
        rows.append([study_id, series_id, npz_filename])
        voxel = np.stack(voxel, -1)
        # voxel = normalize_img(voxel)
        npz_path = output_dir.joinpath(npz_filename)
        np.savez(npz_path, voxel=voxel)
    """

    df = pd.DataFrame(rows, columns=["study_id", "series_id", "filename"])

    if mode == "val":
        split = pd.read_csv("misc/train_with_split.csv")
        df = df.merge(split[["study_id", "fold_id"]], on="study_id", how="left")

    df.to_csv(output_dir.joinpath("metadata.csv"), index=False)


if __name__ == '__main__':
    main()
