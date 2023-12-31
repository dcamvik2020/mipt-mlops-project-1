import os
import sys

import hydra
import pandas as pd
from config import Params
from utils.data import load_data

# from utils.data import save_data
from utils.pipeline import train_model


@hydra.main(config_path="../config_dir", config_name="config", version_base="2.5")
def main(cfg: Params) -> None:
    # print("STEP 1 : save data ... ", end="")
    # save_data("./data/")

    print("STEP 1 : load data", end="")

    X_train, y_train = load_data("./data/", "train.csv")
    train = pd.concat([X_train, y_train], axis=1)
    X_test, y_test = load_data("./data/", "test.csv")
    test = pd.concat([X_test, y_test], axis=1)
    print("OK")
    print("Train data shape:" + str(train.shape))
    print("Test data shape:" + str(test.shape))
    print()

    print("STEP 2 : train & save model ... ", end="")
    # print(f"{cfg['model'] = }")
    model, model_params = train_model(train, cfg["model"], "./models/", "model.pkl")
    print("OK")
    print()

    print("Model hyperparameters:")
    for key in model_params:
        print("    " + key + " = " + str(model_params[key]))
    return 0


if __name__ == "__main__":
    if os.getcwd().split("/")[-1] != "mipt_mlops_project_1":
        os.chdir("./mipt_mlops_project_1")
    sys.exit(main())
