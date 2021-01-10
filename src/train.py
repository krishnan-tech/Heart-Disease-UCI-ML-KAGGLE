import argparse
import os

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

import config
import model_dispatcher

# ignore warnings
from warnings import simplefilter
simplefilter(action='ignore')


def run(fold, model):
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
    classifier = model_dispatcher.models[model]

    # fit the model
    classifier.fit(x_train, y_train)

    # predict the values
    test_preds = classifier.predict(x_valid)
    train_preds = classifier.predict(x_train)

    # roc curve figure
    fpr, tpr, thresholds = roc_curve(y_valid, test_preds)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.rcParams['font.size'] = 12
    plt.title(
        f'ROC curve for {model} classifier - Number {fold+1}/{args.fold+1}')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)
    plt.show()

    # calculate the accuracy
    train_accuracy = metrics.accuracy_score(y_train, train_preds)
    test_accuracy = metrics.accuracy_score(y_valid, test_preds)
    auc_ = auc(fpr, tpr)
    confusion_matrix_value = confusion_matrix(y_valid, test_preds)
    print(confusion_matrix_value)
    print(
        f"Fold={fold}, Test Accuracy={test_accuracy}, Train Accuracy={train_accuracy}, AUC={auc_} Model={model}")
    print("========================================================")

    # save the model
    joblib.dump(classifier, os.path.join(
        config.MODEL_OUTPUT, f"dt_{fold}.bin"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    for fold_ in range(args.fold):
        run(fold=fold_, model=args.model)
