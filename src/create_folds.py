import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection

if __name__ == '__main__':
    # read csv
    df = pd.read_csv('../input/heart.csv')
    # create kfold column
    df['kfold'] = -1
    # randomize data
    df = df.sample(frac=1).reset_index(drop=True)
    # fetch labels
    y = df.target.values
    # init kfolds
    kf = model_selection.StratifiedKFold(n_splits=5)
    # fill the kfold class from model selection
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f
    # save the csv
    df.to_csv('../input/heart_folds.csv', index=True)
