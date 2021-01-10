# LightGBM
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn import metrics

from warnings import simplefilter
simplefilter(action='ignore')

df = pd.read_csv("../input/heart_folds.csv")
for fold in range(4):
    # training data where kfold != fold
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # validation data where kfold == fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # drop the target column and convert it into the array
    x_train = df_train.drop("target", axis=1).values
    y_train = df_train.target.values

    # similarly for validation
    x_valid = df_valid.drop("target", axis=1).values
    y_valid = df_valid.target.values

    # init the LightGBM
    d_train = lgb.Dataset(x_train, label=y_train)
    params = {}
    clf = lgb.train(params, d_train, 100)
    y_pred = clf.predict(x_valid)

    for i in range(0, len(y_pred)):
        if y_pred[i] >= 0.5:       # setting threshold to .5
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    test_accuracy = metrics.accuracy_score(y_valid, y_pred)
    print(test_accuracy)
