import os
import pickle as pkl
import sys

import pandas as pd
from utils.data import load_data
from utils.metrics import metrics, round_metrics
from utils.pipeline import eval_model


def main():
    print("STEP 1 : load test data ... ", end="")
    X_test, y_test = load_data("./data/", "test.csv")
    print("OK")

    print("STEP 2 : load model ... ", end="")
    with open("./models/" + "model.pkl", "rb") as f:
        model = pkl.load(f)
    print("OK")

    print("STEP 3 : save preds ... ", end="")
    test_preds = model.predict(X_test)
    pd.Series(test_preds).to_csv("./data/test_preds.csv")
    print("OK")

    print("STEP 4 : eval model ... ", end="")
    test_data = pd.concat([X_test, y_test], axis=1)
    test_metrics = eval_model(test_data, "./models/", "model.pkl")
    print("OK")
    print()

    print("Test model metrics:")
    for name in test_metrics:
        print("    " + name + " = " + str(test_metrics[name]))
    print()

    print("Test constant (mean) metrics:")
    mean = y_test.mean()
    test_metrics_mean = round_metrics(metrics(y_test, mean), 3)
    for name in test_metrics_mean:
        print("    " + name + " = " + str(test_metrics_mean[name]))
    print()

    print("Test constant (median) metrics:")
    median = y_test.median()
    test_metrics_median = round_metrics(metrics(y_test, median), 3)
    for name in test_metrics_median:
        print("    " + name + " = " + str(test_metrics_median[name]))
    return 0


if __name__ == "__main__":
    if os.getcwd().split("/")[-1] != "mipt_mlops_project_1":
        os.chdir("./mipt_mlops_project_1")
    sys.exit(main())
