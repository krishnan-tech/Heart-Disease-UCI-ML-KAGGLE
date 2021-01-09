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
    "rf": ensemble.RandomForestClassifier(max_depth=5),
    "logistic_regression": LogisticRegression(),
}
