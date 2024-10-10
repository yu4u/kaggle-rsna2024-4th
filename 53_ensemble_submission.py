import argparse
import pandas as pd

level_to_str = {0: "l1_l2", 1: "l2_l3", 2: "l3_l4", 3: "l4_l5", 4: "l5_s1"}


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode", type=str, default="val")  # val or test
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    mode = args.mode
    prefix = "train" if mode == "val" else "test"
    series_df = pd.read_csv(f"input/{prefix}_series_descriptions.csv")

    # create row_ids
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

    axials = [
        (0.15, "eval_ensemble_ax5ch_mask0.1.csv"),
        (0.37, f"yu4u_axial_{mode}_baseline_submission.csv"),
        (0.48, f"yu4u_axial_{mode}_swint_submission.csv"),
    ]

    s1s = [
        (0.21, "eval_ensemble_ax5ch_mask0.1.csv"),
        (0.33, f"yu4u_sagittal1_{mode}_baseline_submission.csv"),
        (0.46, f"yu4u_sagittal1_{mode}_swint_submission.csv"),
    ]

    s2s = [
        (0.21, "eval_ensemble_ax5ch_mask0.1.csv"),
        (0.25, f"yu4u_sagittal2_{mode}_baseline_submission.csv"),
        (0.06, f"yu4u_sagittal2_{mode}_swint_submission.csv"),
        (0.38, f"yu4u_sagittal2_{mode}_axial_submission.csv"),
    ]

    condition_to_csv_filenames = {
        "left_neural_foraminal_narrowing": s1s,
        "right_neural_foraminal_narrowing": s1s,
        "left_subarticular_stenosis": axials,
        "right_subarticular_stenosis": axials,
        "spinal_canal_stenosis": s2s,
    }

    condition_dfs = []

    for condition, row_ids in condition_to_row_ids.items():
        condition_df = pd.DataFrame(row_ids, columns=["row_id"])
        condition_df[["normal_mild", "moderate", "severe"]] = 1e-5
        csv_filenames = condition_to_csv_filenames[condition]

        for w, csv_filename in csv_filenames:
            df = pd.read_csv(csv_filename)
            tmp_df = pd.DataFrame(row_ids, columns=["row_id"])
            tmp_df = tmp_df.merge(df, on="row_id", how="left")
            tmp_df = tmp_df.fillna(0)
            condition_df[["normal_mild", "moderate", "severe"]] += w * tmp_df[["normal_mild", "moderate", "severe"]]

        condition_df[["normal_mild", "moderate", "severe"]] /= condition_df[["normal_mild", "moderate", "severe"]].values.sum(1, keepdims=True)
        condition_dfs.append(condition_df)

    submission_df = pd.concat(condition_dfs, axis=0)
    submission_df = submission_df.sort_values("row_id", ascending=True)
    submission_df.to_csv(f"submission.csv", index=False)


if __name__ == '__main__':
    main()
