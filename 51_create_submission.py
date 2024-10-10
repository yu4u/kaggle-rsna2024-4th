import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch

level_to_str = {0: "l1_l2", 1: "l2_l3", 2: "l3_l4", 3: "l4_l5", 4: "l5_s1"}
axial_suffixes = ["baseline", "swint"]
s1_suffixes = ["baseline", "swint"]
s2_suffixes = ["baseline", "swint", "axial"]


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode", type=str, default="val")  # val or test
    args = parser.parse_args()
    return args



def main():
    args = get_args()
    mode = args.mode

    # axial
    for suffix in axial_suffixes:
        if mode == "val":
            csv_paths = sorted(Path(__file__).parent.glob(f"axial_{mode}_cls_preds_{suffix}_fold*.csv"))
        else:
            csv_paths = [f"axial_{mode}_cls_preds_{suffix}.csv"]

        results = []

        for csv_path in csv_paths:
            print(f"loading {csv_path}")
            df = pd.read_csv(csv_path)

            for study_id, sub_df in df.groupby("study_id"):
                level_to_preds = defaultdict(list)

                for _, row in sub_df.iterrows():
                    level = row["part_id"]
                    pred = row[["left0", "left1", "left2", "right0", "right1", "right2"]].values
                    level_to_preds[level].append(pred)

                for level, preds in level_to_preds.items():
                    preds = np.stack(preds, 0)
                    pred = np.mean(preds, 0).reshape(2, 3)
                    pred = torch.softmax(torch.from_numpy(pred), -1).numpy()

                    for side, side_pred in zip(["left", "right"], pred):
                        row_id = f"{study_id}_{side}_subarticular_stenosis_{level_to_str[level]}"
                        results.append([row_id] + list(side_pred))

        result_df = pd.DataFrame(results, columns=["row_id", "normal_mild", "moderate", "severe"])

        if mode == "val":
            result_df.to_csv(f"yu4u_axial_{suffix}_submission.csv", index=False)
        else:
            result_df.to_csv(f"yu4u_axial_{mode}_{suffix}_submission.csv", index=False)

    # sagittal1
    for suffix in s1_suffixes:
        if mode == "val":
            csv_paths = sorted(Path(__file__).parent.glob(f"sagittal1_{mode}_cls_preds_{suffix}_fold*.csv"))
        else:
            csv_paths = [f"sagittal1_{mode}_cls_preds_{suffix}.csv"]

        results = []

        for csv_path in csv_paths:
            print(f"loading {csv_path}")
            df = pd.read_csv(csv_path)

            for study_id, sub_df in df.groupby("study_id"):
                for side in ["left", "right"]:
                    level_to_preds = defaultdict(list)
                    side_df = sub_df[sub_df["side"] == side]

                    for _, row in side_df.iterrows():
                        level = row["part_id"]
                        pred = row[["left0", "left1", "left2"]].values.astype(np.float32)
                        level_to_preds[level].append(pred)

                    for level, preds in level_to_preds.items():
                        preds = np.stack(preds, 0)
                        pred = np.mean(preds, 0)
                        pred = torch.softmax(torch.from_numpy(pred), -1).numpy()
                        row_id = f"{study_id}_{side}_neural_foraminal_narrowing_{level_to_str[level]}"
                        results.append([row_id] + list(pred))

        result_df = pd.DataFrame(results, columns=["row_id", "normal_mild", "moderate", "severe"])

        if mode == "val":
            result_df.to_csv(f"yu4u_sagittal1_{suffix}_submission.csv", index=False)
        else:
            result_df.to_csv(f"yu4u_sagittal1_{mode}_{suffix}_submission.csv", index=False)

    # sagittal2
    for suffix in s2_suffixes:
        if mode == "val":
            csv_paths = sorted(Path(__file__).parent.glob(f"sagittal2_{mode}_cls_preds_{suffix}_fold*.csv"))
        else:
            csv_paths = [f"sagittal2_{mode}_cls_preds_{suffix}.csv"]

        results = []

        for csv_path in csv_paths:
            print(f"loading {csv_path}")
            df = pd.read_csv(csv_path)

            for study_id, sub_df in df.groupby("study_id"):
                level_to_preds = defaultdict(list)

                for _, row in sub_df.iterrows():
                    level = row["part_id"]
                    pred = row[["left0", "left1", "left2"]].values.astype(np.float32)
                    level_to_preds[level].append(pred)

                for level, preds in level_to_preds.items():
                    preds = np.stack(preds, 0)
                    pred = np.mean(preds, 0)
                    pred = torch.softmax(torch.from_numpy(pred), -1).numpy()
                    row_id = f"{study_id}_spinal_canal_stenosis_{level_to_str[level]}"
                    results.append([row_id] + list(pred))

        result_df = pd.DataFrame(results, columns=["row_id", "normal_mild", "moderate", "severe"])

        if mode == "val":
            result_df.to_csv(f"yu4u_sagittal2_{suffix}_submission.csv", index=False)
        else:
            result_df.to_csv(f"yu4u_sagittal2_{mode}_{suffix}_submission.csv", index=False)



if __name__ == '__main__':
    main()
