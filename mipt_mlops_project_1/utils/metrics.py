import typing as tp

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def round_metrics(metrics: dict, digits: int) -> None:
    """Rounds all metrics before print"""
    for name in metrics:
        metrics[name] = np.round(metrics[name], digits)
    return metrics


def metrics(y_true: np.array, y_pred: tp.Union[np.array, float]) -> tp.Dict[str, float]:
    """Return dict with all needed metrics for preds"""
    if isinstance(y_pred, float):
        y_pred = np.full(fill_value=y_pred, shape=y_true.shape)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }
