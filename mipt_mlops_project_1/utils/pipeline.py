import pickle as pkl
import sys

import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


# from sklearn.model_selection import train_test_split
# from utils.data import load_data, save_data


def train_model(
    data_trn, model_dirname="./models/", model_fname="catboost.pkl", logging=True
):
    """
    try:
        data = load_data(logging=logging)
    except Exception as e:
        save_data(logging=logging)
        data = load_data(logging=logging)

    data_trn, data_tst = train_test_split(
        data, test_size=0.2, random_state=42,
        shuffle=True, stratify=None
    )
    """

    model = CatBoostRegressor(verbose=0)
    feats = list(data_trn.columns[:-1])
    target = data_trn.columns[-1]
    model = model.fit(data_trn[feats], data_trn[target])

    with open(model_dirname + model_fname, "wb") as f:
        pkl.dump(model, f)

    return model


def get_all_metrics(model, X, y):
    return {
        "MAE": np.round(mean_absolute_error(y, model.predict(X)), decimals=5),
        "MAPE": np.round(mean_absolute_percentage_error(y, model.predict(X)), decimals=5),
        "RMSE": np.round(mean_squared_error(y, model.predict(X)), decimals=5),
        "R2": np.round(r2_score(y, model.predict(X)), decimals=5),
    }


def get_all_metrics_const_pred(X, y, pred_value):
    preds = np.full(fill_value=pred_value, shape=y.shape)
    return {
        "MAE": np.round(mean_absolute_error(y, preds), decimals=5),
        "MAPE": np.round(mean_absolute_percentage_error(y, preds), decimals=5),
        "RMSE": np.round(mean_squared_error(y, preds), decimals=5),
        "R2": np.round(r2_score(y, preds), decimals=5),
    }


def eval_model(data_trn, data_tst, model_dirname="./models/", model_fname="catboost.pkl"):
    with open(model_dirname + model_fname, "rb") as f:
        model = pkl.load(f)

    cols = data_trn.columns
    X_trn, y_trn = data_trn[cols[:-1]], data_trn[cols[-1]]
    X_tst, y_tst = data_tst[cols[:-1]], data_tst[cols[-1]]

    scores_trn = get_all_metrics(model, X_trn, y_trn)
    scores_tst = get_all_metrics(model, X_tst, y_tst)

    return scores_trn, scores_tst


def main() -> int:
    model, data_trn, data_tst = train_model()
    scores_trn, scores_tst = eval_model(data_trn, data_tst)
    for metric in ["MAE", "MAPE", "RMSE", "R2"]:
        print("metric:", metric)
        print("Train:", scores_trn[metric])
        print("Test:", scores_tst[metric])
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
