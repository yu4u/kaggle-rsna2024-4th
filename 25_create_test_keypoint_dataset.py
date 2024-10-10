import argparse
from pathlib import Path
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
    root_dir = Path(__file__).parent.joinpath("input")
    output_dir = Path(f"axial_keypoint_{mode}_dataset")
    output_dir.mkdir(exist_ok=True)
    rows = []
    # val
    # axial_val_preds_fold*.npz
    # test
    # axial_test_preds.npz

    if mode == "val":
        npz_paths = sorted(Path(__file__).parent.glob(f"axial_{mode}_preds_fold*.npz"))
    else:
        npz_paths = [Path(__file__).parent.joinpath(f"axial_{mode}_preds.npz")]

    for npz_path in npz_paths:
        filename_to_preds = np.load(npz_path)

        for filename in tqdm(filename_to_preds["filenames"]):
            study_id, series_id = [int(x) for x in filename.replace(".npz", "").split("_")]
            sub_preds = filename_to_preds[filename]
            instance_numbers = filename_to_preds[f"{filename}_instance_numbers"]
            diff = sub_preds[2:, 1:] - sub_preds[:-2, 1:] - (sub_preds[2:, :-1] - sub_preds[:-2, :-1])
            diff = np.abs(diff)

            for part_id, (d_i, d_score) in enumerate(zip(diff.argmax(0), diff.max(0))):
                preds_max = sub_preds.max(0)

                if preds_max[part_id] > 0.5 and preds_max[part_id + 1] > 0.5:
                    instance_number = instance_numbers[d_i + 1]
                    prefix = "train" if mode == "val" else "test"
                    dcm_paths = sorted(Path(f"input/{prefix}_images/{study_id}/{series_id}").glob("*.dcm"),
                                       key=lambda x: int(x.stem))
                    # get index of current instance_number
                    idx = [int(p.stem) for p in dcm_paths].index(instance_number)
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
                    img_filename = f"{study_id}_{series_id}_{instance_number}_{part_id}_img.npy"
                    output_img_path = output_dir.joinpath(img_filename)
                    np.save(output_img_path, img)
                    rows.append([study_id, series_id, img_filename, part_id])

    df = pd.DataFrame(rows, columns=["study_id", "series_id", "filename", "part_id"])

    if mode == "val":
        split = pd.read_csv("misc/train_with_split.csv")
        df = df.merge(split[["study_id", "fold_id"]], on="study_id", how="left")

    df.to_csv(output_dir.joinpath("metadata.csv"), index=False)


if __name__ == '__main__':
    main()
