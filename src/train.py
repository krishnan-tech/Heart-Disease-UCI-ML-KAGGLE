import argparse
import os

import joblib
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

import config
import model_dispatcher

# ignore warnings
from warnings import simplefilter
simplefilter(action='ignore')


def run(fold):
    # read the training data with folds
    df = pd.read_csv(config.TRAINING_FILE)
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

    # init the simple descision tree
    clf = LogisticRegression(random_state=0)

    # fit the model
    clf.fit(x_train, y_train)

    # predict the values
    preds = clf.predict(x_valid)

    # calculate the accuracy
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")

    # save the model
    joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    for fold_ in range(args.fold):
        run(fold=fold_)
