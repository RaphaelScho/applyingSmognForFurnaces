import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

from utils import run_smogn
from config import data_path, resample_path


def cross_validate(df, model_params, use_smogn, folds=10, rel_thresh=0.1):
    X = df.drop("ignitions", axis=1)
    y = df.ignitions

    kf = KFold(n_splits=folds)
    scores = []

    for fold, (train_index, test_index) in enumerate(kf.split(X, y), 1):
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]

        model = RandomForestRegressor(**model_params)

        if use_smogn:
            input = X_train.copy()
            input["ignitions"] = y_train
            run_smogn(data_path, parallel=True, rel_thresh=rel_thresh, silent=True, features=input)
            df_resampled = pd.read_pickle(resample_path)

            X_train_oversampled = df_resampled.drop("ignitions", axis=1)
            y_train_oversampled = df_resampled.ignitions

            model.fit(X_train_oversampled, y_train_oversampled)

        else:
            model.fit(X_train, y_train)

        scores.append(model.score(X_test, y_test))  # R2

    return scores