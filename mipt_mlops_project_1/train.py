import os
import sys

import pandas as pd

from utils.data import save_data, load_data
from utils.pipeline import train_model, eval_model

from utils.logging import logging_print

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


    logging = (logging == "yes")
    # logging_low = 1 - logging # if logging, only high-level logging
    logging_low = False
    
    #______________________________________
    
    logging_print(logging=logging, line="STEP 1 : load & save data ... ", end="")
    
    save_data(path_to_save=data_dir, fname=data_file, logging=logging_low)
    data_trn, data_tst = load_data(load_path=data_dir, fname=data_file, logging=logging_low)    
    data = pd.concat([data_trn, data_tst], axis=0)

    # dms = data_memory_size
    dms = data.memory_usage(index=True, deep=True).sum()
    gb = dms // (1024 ** 3)
    mb = (dms % (1024 ** 3)) // (1024 ** 2)
    kb = (dms % (1024 ** 2)) // 1024
    memory_str = f"{gb} GB, {mb} MB, {kb} KB"

    logging_print(logging=logging, line="OK")
    logging_print(logging=logging, line="Data shape:" + str(data.shape))
    logging_print(logging=logging, line="Data memory usage:" + memory_str)
    logging_print(logging=logging, line="-----------------------------------------")
    
    #______________________________________
    
    logging_print(logging=logging, line="STEP 2 : train & save model ... ", end="")
    model = train_model(data_trn, model_dirname=model_dir, model_fname=model_file, logging=logging_low)
    logging_print(logging=logging, line="OK")
    logging_print(logging=logging, line="-----------------------------------------")
    
    return 0

if __name__ == '__main__':
    print("Current working directory:", os.getcwd())
    if os.getcwd().split('/')[-1] != 'mipt-mlops-course':
        print("Move to project directory...", end='')
        os.chdir('./mipt-mlops-course')
        print("OK")
        print("Current working directory:", os.getcwd())

    sys.exit(main())
