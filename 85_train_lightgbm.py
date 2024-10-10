import lightgbm as lgb
from sklearn.metrics import log_loss
import numpy as np
import pandas as pd

target_to_params = {
    "spinal_canal_stenosis": {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "verbosity": -1,
        "n_jobs": -1,
        'boosting_type': 'gbdt',
        'random_state': 42,
        "learning_rate": 0.02,
        "llambda_l1": 1.8924112498154402e-06,
        "lambda_l2": 9.868704968397868,
        "min_split_gain": 0.2443881418427369,
        "min_data_in_leaf": 5,
        "max_depth": 3,
        "bagging_fraction": 0.5365431789365274,
        "feature_fraction": 0.5004083005149835,
        "bagging_freq": 9,
    },
    "neural_foraminal_narrowing": {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "verbosity": -1,
        "n_jobs": -1,
        'boosting_type': 'gbdt',
        'random_state': 42,
        "learning_rate": 0.02,
        "lambda_l1": 5.1698395330303e-08,
        "lambda_l2": 1.3900677016206051,
        "min_split_gain": 7.178072179354015e-08,
        "min_data_in_leaf": 3,
        "max_depth": 3,
        "bagging_fraction": 0.5181120504302912,
        "feature_fraction": 0.5591070404973719,
        "bagging_freq": 1,
    },
    "subarticular_stenosis": {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "verbosity": -1,
        "n_jobs": -1,
        'boosting_type': 'gbdt',
        'random_state': 42,
        "learning_rate": 0.02,
        "lambda_l1": 5.1634373343386724e-05,
        "lambda_l2": 4.220919582104601e-06,
        "min_split_gain": 2.6835672485371036e-08,
        "min_data_in_leaf": 9,
        "max_depth": 3,
        "bagging_fraction": 0.6021159271545301,
        "feature_fraction": 0.5085686764211453,
        "bagging_freq": 9,
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

        df["level"] = df["level"].astype("category")
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

            train_data = lgb.Dataset(X_train, label=y_train, weight=weights_train,
                                     categorical_feature=["level"])
            val_data = lgb.Dataset(X_val, label=y_val, weight=weights_val,
                                   categorical_feature=["level"])
            params = target_to_params[target]
            model = lgb.train(params,
                              train_data,
                              num_boost_round=50000,
                              valid_sets=[train_data, val_data],
                              callbacks=[lgb.early_stopping(stopping_rounds=500,
                                                            verbose=True)]
                              )

            y_pred = model.predict(X_val)
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

            checkpoint_name = f"yu4u_{target}_stacking_fold_{fold_id}.txt"
            model.save_model(checkpoint_name)

        print(f"Mean Validation Weighted Log Loss: {np.mean(val_losses)}")
        result_df = pd.DataFrame(results, columns=["row_id", "normal_mild", "moderate", "severe"])
        result_df.to_csv(f"yu4u_{target}_stacking_submission.csv", index=False)


if __name__ == "__main__":
    main()
