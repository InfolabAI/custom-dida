import dataprep
import pandas as pd
import numpy as np
from model_DIDA.utils.data_util import *


# create function to load data and to input dataprep
def dataprep():
    # load data
    from config import args

    args, data = load_data(args)
    # use dataprep for numpy array from data

    # split data into train and test
    train = data[data["train"] == 1]
    test = data[data["train"] == 0]

    # split train and test into X and y
    X_train = train.drop(columns=["train", "target"])
    y_train = train["target"]
    X_test = test.drop(columns=["train", "target"])
    y_test = test["target"]

    # input data into dataprep
    train_data = load_data.DataPrep(X_train, y_train)
    test_data = load_data.DataPrep(X_test, y_test)

    # return train and test data
    return train_data, test_data


if __name__ == "__main__":
    train_data, test_data = dataprep()
    breakpoint()
