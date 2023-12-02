import typing as tp

import dvc.api
import pandas as pd
from sklearn.datasets import load_boston


TRAIN_RATIO = 0.7


def save_data(path_to_save: str = "./data/") -> None:
    data = load_boston()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    data = pd.concat([X, y], axis=1)
    train_size = int(data.shape[0] * TRAIN_RATIO)
    train, test = data[:train_size], data[train_size:]
    train.to_csv(path_to_save + "train.csv", index=False)
    test.to_csv(path_to_save + "test.csv", index=False)


def load_data(
    load_path: str = "./data/", fname: str = "train.csv"
) -> tp.Tuple[pd.DataFrame, pd.Series]:
    with dvc.api.open(
        # './mipt_mlops_project_1/' +
        load_path + fname,
        remote="my_remote_storage",
    ) as f:
        data = pd.read_csv(f)
    # data = pd.read_csv(load_path + fname)
    X = data[data.columns[:-1]]
    y = data[data.columns[-1]]
    return X, y
