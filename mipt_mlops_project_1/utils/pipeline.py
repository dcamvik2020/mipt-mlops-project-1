import pickle as pkl

import pandas as pd
from catboost import CatBoostRegressor

from .metrics import metrics, round_metrics


def train_model(
    train: pd.DataFrame, save_path: str = "./models/", fname: str = "model.pkl"
) -> CatBoostRegressor:
    features, target_col = list(train.columns[:-1]), train.columns[-1]
    model_params = {"iterations": 50, "max_depth": 5, "learning_rate": 0.5}
    model = CatBoostRegressor(**model_params, silent=True, random_seed=8)
    model.fit(train[features], train[target_col])
    with open(save_path + fname, "wb") as f:
        pkl.dump(model, f)
    return model, model_params


def eval_model(
    data: pd.DataFrame, model_dir: str = "./models/", fname: str = "model.pkl"
) -> dict:
    """Return dict with all needed metrics for model"""
    with open(model_dir + fname, "rb") as f:
        model = pkl.load(f)
    features, target_col = data.columns[:-1], data.columns[-1]
    X, y = data[features], data[target_col]
    return round_metrics(metrics(y, model.predict(X)), 3)
