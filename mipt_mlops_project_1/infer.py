import os
import pickle as pkl
import sys

import pandas as pd
from utils.data import load_data
from utils.logging import logging_print
from utils.pipeline import eval_model


def main():
    """
    while True:
        logging = input("Enable logging (yes/no): ")
        if logging in {"yes", "no"}:
            break

    data_dir = input("Save data to directory: ")
    if data_dir == "":
        data_dir = './data/'

    data_file = input("Save data to file: ")
    if data_file == "":
        data_file = "tmp_data.csv"

    model_dir = input("Save model to directory: ")
    if model_dir == "":
        model_dir = './models/'

    model_file = input("Save model to file: ")
    if model_file == "":
        model_file = "catboost.pkl"
    """

    logging = "yes"
    data_dir = "./data/"
    data_file = "tmp_data.csv"
    model_dir = "./models/"
    model_file = "model.pkl"

    logging = logging == "yes"
    # logging_low = 1 - logging # if logging, only high-level logging
    logging_low = False

    # ______________________________________

    logging_print(logging=logging, line="STEP 1 : load data ... ", end="")

    data_trn, data_tst = load_data(
        load_path=data_dir, fname=data_file, logging=logging_low
    )

    logging_print(logging=logging, line="OK")
    logging_print(logging=logging, line="-----------------------------------------")

    # ______________________________________

    logging_print(logging=logging, line="STEP 2 : load model ... ", end="")
    with open(model_dir + model_file, "rb") as f:
        model = pkl.load(f)
    logging_print(logging=logging, line="OK")
    logging_print(logging=logging, line="-----------------------------------------")

    # ______________________________________

    logging_print(logging=logging, line="STEP 3 : eval model, save preds ... ", end="")
    scores_trn, scores_tst = eval_model(
        data_trn, data_tst, model_dirname=model_dir, model_fname=model_file
    )
    preds = model.predict(data_tst[data_tst.columns[:-1]])
    preds = pd.Series(preds)
    preds.to_csv("tmp_preds.csv")

    logging_print(logging=logging, line="OK")
    logging_print(logging=logging, line="Train:" + str(scores_trn))
    logging_print(logging=logging, line="Test:" + str(scores_tst))

    return 0


if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
    if os.getcwd().split("/")[-1] != "mipt_mlops_project_1":
        print("Move to project directory...", end="")
        os.chdir("./mipt_mlops_project_1")
        print("OK")
        print("Current working directory:", os.getcwd())

    sys.exit(main())
