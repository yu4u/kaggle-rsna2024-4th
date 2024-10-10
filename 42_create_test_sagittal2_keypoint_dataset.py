import argparse
from pathlib import Path
import pydicom
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

from axial_src.util import get_img_from_dcm, normalize_img


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--mode", type=str, default="val")  # val or test
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    img_size = args.img_size
    mode = args.mode  # val or test
    prefix = "train" if mode == "val" else "test"
    root_dir = Path(__file__).parent.joinpath("input")
    output_dir = Path(f"sagittal2_keypoint_{mode}_dataset")
    output_dir.mkdir(exist_ok=True)
    rows = []
    df = pd.read_csv(f"input/{prefix}_series_descriptions.csv")
    df = df[df["series_description"] == "Sagittal T2/STIR"]

    for _, row in tqdm(df.iterrows(), total=len(df)):
        study_id = row["study_id"]
        series_id = row["series_id"]
        dcm_paths = sorted(Path(f"input/{prefix}_images/{study_id}/{series_id}").glob("*.dcm"),
                           key=lambda x: int(x.stem))
        # get index of current instance_number
        idx = len(dcm_paths) // 2
        instance_number = int(dcm_paths[idx].stem)
        imgs = []

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
        img = cv2.resize(img, (img_size, img_size))
        # img = normalize_img(img)
        img_filename = f"{study_id}_{series_id}_{instance_number}_img.npy"
        output_img_path = output_dir.joinpath(img_filename)
        np.save(output_img_path, img)
        rows.append([study_id, series_id, img_filename])

    df = pd.DataFrame(rows, columns=["study_id", "series_id", "filename"])

    if mode == "val":
        split = pd.read_csv("misc/train_with_split.csv")
        df = df.merge(split[["study_id", "fold_id"]], on="study_id", how="left")

    df.to_csv(output_dir.joinpath("metadata.csv"), index=False)


if __name__ == '__main__':
    main()
