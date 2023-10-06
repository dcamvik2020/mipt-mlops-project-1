import pandas as pd
from sklearn.datasets import load_boston
from utils.logging import logging_print


def save_data(path_to_save="./data/", fname="tmp_data.csv", logging=True):
    logging_print(logging=logging, line=f"save data to {path_to_save + fname}")
    logging_print(logging=logging, line="START ...")

    data = load_boston()

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    data = pd.concat([X, y], axis=1)
    data.to_csv(path_to_save + fname, index=False)

    logging_print(logging=logging, line="SUCCESS!")
    logging_print(logging=logging, line="Data location:" + path_to_save + fname)
    logging_print(logging=logging)


def load_data(load_path="./data/", fname="tmp_data.csv", logging=True):
    logging_print(logging=logging, line=f"load data from {load_path + fname}")
    logging_print(logging=logging, line="START ...")

    data = pd.read_csv(load_path + fname)

    # dms = data_memory_size
    dms = data.memory_usage(index=True, deep=True).sum()
    gb = dms // (1024**3)
    mb = (dms % (1024**3)) // (1024**2)
    kb = (dms % (1024**2)) // 1024
    memory_str = f"{gb} GB, {mb} MB, {kb} KB"

    logging_print(logging=logging, line="SUCCESS!")
    logging_print(logging=logging, line="Data shape:" + str(data.shape))
    logging_print(logging=logging, line="Data memory usage:" + memory_str)
    logging_print(logging=logging)

    # return train & test, test = last 100 objects
    return data[:-100], data[-100:]
