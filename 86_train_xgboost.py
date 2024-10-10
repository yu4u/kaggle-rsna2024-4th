import xgboost as xgb
from sklearn.metrics import log_loss
import numpy as np
import pandas as pd

target_to_params = {
    "spinal_canal_stenosis": {
        "objective": "multi:softprob",
        "num_class": 3,
        "metric": "multi_logloss",
        "verbosity": 0,
        "n_jobs": -1,
        'random_state': 42,
        "tree_method": "hist",
        "max_depth": 3,
        "min_child_weight": 6.105403689236205,
        "gamma": 0.0036139141070800763,
        "subsample": 0.6439745721326602,
        "alpha": 0.008333741182808168,
        "lambda": 0.003650799141525395,
        "colsample_bytree": 0.5270484715396705,
        "colsample_bylevel": 0.5005413332572565,
        "colsample_bynode": 0.5000348634644112,
        "eta": 0.001,
    },
    "neural_foraminal_narrowing": {
        "objective": "multi:softprob",
        "num_class": 3,
        "metric": "multi_logloss",
        "verbosity": 0,
        "n_jobs": -1,
        'random_state': 42,
        "tree_method": "hist",
        "max_depth": 3,
        "min_child_weight": 9.019054990957335,
        "gamma": 0.36480310224303747,
        "subsample": 0.9818076642958057,
        "alpha": 0.49674752026998564,
        "lambda": 4.063614279527356,
        "colsample_bytree": 0.6697256050727396,
        "colsample_bylevel": 0.6099551495104498,
        "colsample_bynode": 0.5168176418964282,
        "eta": 0.001,
    },
    "subarticular_stenosis": {
        "objective": "multi:softprob",
        "num_class": 3,
        "metric": "multi_logloss",
        "verbosity": 0,
        "n_jobs": -1,
        'random_state': 42,
        "tree_method": "hist",
        "max_depth": 3,
        "min_child_weight": 9.173767571278209,
        "gamma": 0.019076894273373314,
        "subsample": 0.9996068708115513,
        "alpha": 2.2143896638020195e-08,
        "lambda": 9.526215281187552,
        "colsample_bytree": 0.5205444050133367,
        "colsample_bylevel": 0.5756240561019075,
        "colsample_bynode": 0.5323571678221571,
        "eta": 0.001,
    }
}


def main():
    for target in ["spinal_canal_stenosis", "neural_foraminal_narrowing", "subarticular_stenosis"]:
        gt_df_original = pd.read_csv("stacking_gt.csv")

        if target == "spinal_canal_stenosis":
            df = pd.read_csv(f"stacking_features_spinal_canal_stenosis.csv")
            gt_df = gt_df_original[target]
        else:
            dfs = []
            for side in ["left", "right"]:
                df = pd.read_csv(f"stacking_features_{side}_{target}.csv")
                dfs.append(df)
            df = pd.concat(dfs)

            gt_dfs = []
            for side in ["left", "right"]:
                gt_dfs.append(gt_df_original[f"{side}_{target}"])
            gt_df = pd.concat(gt_dfs)

        df["level"] = df["level"].map({"l1_l2": 0, "l2_l3": 1, "l3_l4": 2, "l4_l5": 3, "l5_s1": 4})
        class_weights = {0: 1, 1: 2, 2: 4, -100: 0}
        val_losses = []
        results = []

        for fold_id in range(5):
            print(f"Fold {fold_id}")
            X_train = df[df["fold_id"] != fold_id]
            y_train = gt_df[df["fold_id"] != fold_id].values
            X_val = df[df["fold_id"] == fold_id]
            y_val = gt_df[df["fold_id"] == fold_id].values
            study_ids = X_val["study_id"].values
            levels = X_val["level"].values
            del X_train["study_id"]
            del X_val["study_id"]
            del X_train["fold_id"]
            del X_val["fold_id"]

            weights_train = np.array([class_weights[y] for y in y_train])
            weights_val = np.array([class_weights[y] for y in y_val])

            y_train[y_train == -100] = 0
            y_val[y_val == -100] = 0

            d_train = xgb.DMatrix(X_train, label=y_train, weight=weights_train)
            d_val = xgb.DMatrix(X_val, label=y_val, weight=weights_val)
            params = target_to_params[target]
            num_round = 10000
            model = xgb.train(params,
                              d_train,
                              num_round,
                              evals=[(d_val, 'test')],
                              early_stopping_rounds=1000,
                              verbose_eval=False
                              )

            y_pred = model.predict(d_val)
            val_loss = log_loss(y_val, y_pred, sample_weight=weights_val, labels=[0, 1, 2])
            print(f"Validation Weighted Log Loss: {val_loss}\n")
            val_losses.append(val_loss)

            for i, (study_id, level, pred) in enumerate(zip(study_ids, levels, y_pred)):
                if target == "spinal_canal_stenosis":
                    row_id = f"{study_id}_{target}_{level}"
                else:
                    side = "left" if i < len(study_ids) // 2 else "right"
                    row_id = f"{study_id}_{side}_{target}_{level}"

                results.append([row_id, pred[0], pred[1], pred[2]])

            checkpoint_name = f"yu4u_{target}_xgboost_stacking_fold_{fold_id}.json"
            model.save_model(checkpoint_name)

        print(f"Mean Validation Weighted Log Loss: {np.mean(val_losses)}")
        result_df = pd.DataFrame(results, columns=["row_id", "normal_mild", "moderate", "severe"])
        result_df.to_csv(f"yu4u_{target}_xgboost_stacking_submission.csv", index=False)


if __name__ == "__main__":
    main()
