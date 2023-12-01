import os

import fire
import infer as infer_module
import train as train_module


def train():
    if os.getcwd().split("/")[-1] != "mipt_mlops_project_1":
        os.chdir("./mipt_mlops_project_1")
    train_module.main()


def infer():
    if os.getcwd().split("/")[-1] != "mipt_mlops_project_1":
        os.chdir("./mipt_mlops_project_1")
    infer_module.main()


if __name__ == "__main__":
    fire.Fire()
