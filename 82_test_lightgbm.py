import argparse
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    for target in ["neural_foraminal_narrowing", "spinal_canal_stenosis", "subarticular_stenosis"]:
        if target == "spinal_canal_stenosis":
            df = pd.read_csv(f"stacking_features_spinal_canal_stenosis.csv")
        else:
            dfs = []
            for side in ["left", "right"]:
                df = pd.read_csv(f"stacking_features_{side}_{target}.csv")
                dfs.append(df)
            df = pd.concat(dfs)

        df["level"] = df["level"].astype("category")

        X_val = df
        study_ids = X_val["study_id"].values
        levels = X_val["level"].values
        del X_val["study_id"]

        preds = []

        for model_file in Path(args.checkpoint_dir).glob(f"yu4u_{target}_stacking_fold_*.txt"):
            print(f"Loading {model_file}")
            model = lgb.Booster(model_file=str(model_file))
            y_pred = model.predict(X_val)
            preds.append(y_pred)

        y_pred = np.mean(preds, axis=0)

        results = []

        for i, (study_id, level, pred) in enumerate(zip(study_ids, levels, y_pred)):
            if target == "spinal_canal_stenosis":
                row_id = f"{study_id}_{target}_{level}"
            else:
                side = "left" if i < len(study_ids) // 2 else "right"
                row_id = f"{study_id}_{side}_{target}_{level}"

            results.append([row_id, pred[0], pred[1], pred[2]])

        result_df = pd.DataFrame(results, columns=["row_id", "normal_mild", "moderate", "severe"])
        result_df.to_csv(f"yu4u_{target}_test_stacking_submission.csv", index=False)


if __name__ == "__main__":
    main()
