# model_dispatcher.py
from sklearn import tree
from sklearn import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

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
    "svm":  SVC(kernel='rbf'),
    "naive_bayes": GaussianNB(),
    "xgboost":  XGBClassifier(),
    "knn": KNeighborsClassifier(n_neighbors=2),
    # Weight 2 for logistic regression to avaoid overfitting in random forest
    "votingclassifier": ensemble.VotingClassifier(estimators=[('lr', LogisticRegression()), ('rf', ensemble.RandomForestClassifier(criterion="entropy", max_depth=1, n_estimators=300)), ('gnb', XGBClassifier())], voting='soft', weights=[2, 1, 1], flatten_transform=True)
}
