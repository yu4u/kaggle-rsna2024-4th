import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode", type=str, default="val")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    mode = args.mode
    prefix = "train" if mode == "val" else "test"
    series_df = pd.read_csv(f"input/{prefix}_series_descriptions.csv")
    condition_to_row_ids = dict()

    conditions = [
        "left_neural_foraminal_narrowing",
        "left_subarticular_stenosis",
        "right_neural_foraminal_narrowing",
        "right_subarticular_stenosis",
        "spinal_canal_stenosis",
    ]

    levels = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]

    for condition in conditions:
        row_ids = []
        row_study_ids = []
        row_levels = []

        for study_id in series_df["study_id"].unique():
            for level in levels:
                row_id = f"{study_id}_{condition}_{level}"
                row_ids.append(row_id)
                row_study_ids.append(study_id)
                row_levels.append(level)

        condition_to_row_ids[condition] = row_ids

    if mode == "val":
        oof_dir = Path("oofs")

        tattaka = [
            #"eval_ensemble_ax5ch_mask0.1.csv",
            "eval_caformer_s18_ax5ch_mask0.1.csv",
            "eval_resnetrs50_ax5ch_mask0.1.csv",
            "eval_swinv2_tiny_ax5ch_mask0.1.csv",
            "eval_rdnet_tiny_ax5ch_mask0.1.csv",
            "eval_maxxvitv2_nano_ax5ch_mask0.1.csv",
        ]

        yu4u_axial = [
            "yu4u_axial_baseline_submission.csv",
            "yu4u_axial_swint_submission.csv",
        ]

        yu4u_s1 = [
            "yu4u_sagittal1_baseline_submission.csv",
            "yu4u_sagittal1_s1_swint_submission.csv",
        ]

        yu4u_s2 = [
            "yu4u_sagittal2_baseline_submission.csv",
            "yu4u_sagittal2_swint_submission.csv",
            "yu4u_sagittal2_axial_submission.csv",
        ]
    else:
        oof_dir = Path("/kaggle/working/")

        tattaka = [
            "tattaka_caformer_s18_20x1x128x128_x1.0_20x1x128x128_x1.0_5x1x128x128_x1.0_mixup_mask0.1.csv",
            # "eval_caformer_s18_ax5ch_mask0.1.csv",
            "tattaka_resnetrs50_20x1x128x128_x1.0_20x1x128x128_x1.0_5x1x128x128_x1.0_mixup_mask0.1.csv",
            # "eval_resnetrs50_ax5ch_mask0.1.csv",
            "tattaka_swinv2_tiny_20x1x128x128_x1.0_20x1x128x128_x1.0_5x1x128x128_x1.0_mixup_mask0.1.csv",
            # "eval_swinv2_tiny_ax5ch_mask0.1.csv",
            "tattaka_rdnet_tiny_30x1x128x128_x1.0_30x1x128x128_x1.0_5x1x128x128_x1.0_mixup_mask0.1.csv",
            # "eval_rdnet_tiny_ax5ch_mask0.1.csv",
            "tattaka_maxxvitv2_nano_30x1x128x128_x1.0_30x1x128x128_x1.0_5x1x128x128_x1.0_mixup_mask0.1.csv",
            # "eval_maxxvitv2_nano_ax5ch_mask0.1.csv",
        ]

        yu4u_axial = [
            "yu4u_axial_test_baseline_submission.csv",
            "yu4u_axial_test_swint_submission.csv",
        ]

        yu4u_s1 = [
            "yu4u_sagittal1_test_baseline_submission.csv",
            "yu4u_sagittal1_test_swint_submission.csv",
        ]

        yu4u_s2 = [
            "yu4u_sagittal2_test_baseline_submission.csv",
            "yu4u_sagittal2_test_swint_submission.csv",
            "yu4u_sagittal2_test_axial_submission.csv",
        ]

    condition_to_csv_filenames = {
        "left_neural_foraminal_narrowing": tattaka + yu4u_s1,
        "right_neural_foraminal_narrowing": tattaka + yu4u_s1,
        "left_subarticular_stenosis": tattaka + yu4u_axial,
        "right_subarticular_stenosis": tattaka + yu4u_axial,
        "spinal_canal_stenosis": tattaka + yu4u_s2,
    }

    condition_dfs = []

    for condition, row_ids in condition_to_row_ids.items():
        condition_df = pd.DataFrame(row_ids, columns=["row_id"])
        csv_filenames = condition_to_csv_filenames[condition]

        for csv_filename in csv_filenames:
            csv_path = oof_dir / csv_filename
            df = pd.read_csv(csv_path)
            print(f"loading {csv_path}", df.shape)
            condition_df = condition_df.merge(df, on="row_id", how="left", suffixes=("", f"_{csv_path.stem}"))

        condition_dfs.append(condition_df)

    split = pd.read_csv("misc/train_with_split.csv")

    for condition, condition_df in zip(conditions, condition_dfs):
        feature = condition_df.iloc[:, 1:].values
        df = pd.DataFrame(feature)
        df["study_id"] = row_study_ids
        df["level"] = row_levels

        if mode == "val":
            df = df.merge(split[["study_id", "fold_id"]], on="study_id", how="left")

        df.to_csv(f"stacking_features_{condition}.csv", index=False)

    if mode == "val":
        # only for val
        train_df = pd.read_csv("input/train.csv", index_col=0)
        gts = []

        def str_to_class_id(s):
            if s == "Normal/Mild":
                return 0
            elif s == "Moderate":
                return 1
            elif s == "Severe":
                return 2
            elif np.isnan(s):
                return -100
            else:
                raise ValueError(f"Invalid class: {s}")

        for study_id in series_df["study_id"].unique():
            for level in levels:
                columns = [f"{condition}_{level}" for condition in conditions]
                gt = train_df.loc[study_id, columns].values
                gts.append([str_to_class_id(s) for s in gt])

        gt_df = pd.DataFrame(gts, columns=conditions)
        gt_df.to_csv("stacking_gt.csv", index=False)


if __name__ == '__main__':
    main()
