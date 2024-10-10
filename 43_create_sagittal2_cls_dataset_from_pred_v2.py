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
    parser.add_argument("--dirname", type=str, default="sagittal2_cls_all_dataset")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--keypoint_img_size", type=int, default=512)
    parser.add_argument("--scale_rate", type=float, default=0.5)
    parser.add_argument("--mode", type=str, default="val")  # val or test
    parser.add_argument("--channel_num", type=int, default=5)
    args = parser.parse_args()
    return args


def calculate_distances(points):
    diffs = points[1:] - points[:-1]
    distances = np.linalg.norm(diffs, axis=1)
    return distances


def crop_img(img, x, y, scale):
    h, w = img.shape[:2]
    x1 = max(0, int(x - scale))
    y1 = max(0, int(y - scale))
    x2 = min(w, int(x + scale))
    y2 = min(h, int(y + scale))
    img = img[y1:y2, x1:x2]
    return img


def main():
    args = get_args()
    img_size = args.img_size
    mode = args.mode  # val or test
    channel_num = args.channel_num
    root_dir = Path(__file__).parent.joinpath("input")

    if mode == "val":
        dfs = [pd.read_csv(f"sagittal2_val_keypoint_preds_fold{i}.csv") for i in range(5)]
        df = pd.concat(dfs, axis=0)
        output_dir = root_dir.joinpath(args.dirname)
    else:
        df = pd.read_csv("sagittal2_test_keypoint_preds.csv")
        output_dir = Path(f"sagittal2_cls_test_dataset")

    train_df = pd.read_csv("misc/train_with_split.csv")
    output_dir.mkdir(exist_ok=True)
    rows = []

    for series_id, sub_df in tqdm(df.groupby("series_id")):
        original_keypoints = []

        for row in sub_df.itertuples(index=False):
            study_id, series_id, instance_number, part_id, x, y = row
            original_keypoints.append((x, y))

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
            # img = get_img_from_dcm(dcm_path, normalize=True, to_uint8=True)
            imgs.append(img)

        for i in range(channel_num):
            if imgs[i].shape != imgs[channel_num // 2].shape:
                imgs[i] = imgs[channel_num // 2].copy()

        img = np.stack(imgs, -1)
        h, w = img.shape[:2]
        keypoints = []

        for x, y in original_keypoints:
            x = x * w / args.keypoint_img_size
            y = y * h / args.keypoint_img_size
            keypoints.append((x, y))

        distances = calculate_distances(np.array(keypoints))
        scale = distances.mean() * args.scale_rate

        """
        img = img[:, :, 1:4].copy()
        for x, y in keypoints:
            cv2.rectangle(img, (int(x - scale), int(y - scale)), (int(x + scale), int(y + scale)), (0, 255, 0), 2)

        cv2.imshow("img", img)
        cv2.waitKey(-1)
        """

        for part_id, (x, y) in enumerate(keypoints):
            li_to_level = {0: "l1_l2", 1: "l2_l3", 2: "l3_l4", 3: "l4_l5", 4: "l5_s1"}
            level = li_to_level[part_id]

            if mode == "val":
                target_column_name = f"spinal_canal_stenosis_{level}"
                target = train_df[train_df["study_id"] == study_id][target_column_name].values[0]
            else:
                target = "Normal/Mild"

            cropped_img = crop_img(img, x, y, scale)
            filename = f"{study_id}_{series_id}_{part_id}_{instance_number}.npz"
            output_img_path = output_dir.joinpath(filename)
            cropped_img = normalize_img(cropped_img)
            cropped_img = cropped_img * 255
            np.savez(output_img_path, img=cropped_img)
            rows.append([study_id, series_id, filename, instance_number, level, target])

    df = pd.DataFrame(rows, columns=["study_id", "series_id", "filename", "instance_number", "level", "target"])
    df = df.merge(train_df[["study_id", "fold_id"]], on="study_id", how="left")
    df.to_csv(output_dir.joinpath("metadata.csv"), index=False)


if __name__ == '__main__':
    main()
