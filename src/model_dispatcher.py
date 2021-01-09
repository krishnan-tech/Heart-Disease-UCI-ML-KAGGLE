# model_dispatcher.py
from sklearn import tree
from sklearn import ensemble
from sklearn.linear_model import LogisticRegression
models = {
    "decision_tree_gini": tree.DecisionTreeClassifier(
        criterion="gini"
    ),
    "decision_tree_entropy": tree.DecisionTreeClassifier(
        criterion="entropy"
    ),
    "rf": ensemble.RandomForestClassifier(
        # around 80 to 90
        criterion="gini", max_depth=7, n_estimators=200
        # arouond 80 to 85
        # criterion="entropy", max_depth=1, n_estimators=300
    ),
    "logistic_regression": LogisticRegression(),
}
