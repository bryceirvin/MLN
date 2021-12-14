import numpy as np
import os
import pandas as pd

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, recall_score, precision_score

# use sklearn models as baseline
# calling these without arguments will result in default settings

def get_knn_regressor(n_neighbors=5, weights="uniform", algorithm="auto",
                  leaf_size=30, p=2):

    return KNeighborsRegressor(n_neighbors, weights, algorithm, leaf_size, p)


def get_knn_classifier(n_neighbors=5, weights="uniform", algorithm="auto",
                  leaf_size=30, p=2):
    
    return KNeighborsClassifier(n_neighbors, weights, algorithm, leaf_size, p)


def get_svm_classifier(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False,
                       tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1,
                       decision_function_shape='ovr', break_ties=False, random_state=None):

    return SVC(C, kernel, degree, gamma, coef0, shrinking, probability, tol, cache_size, class_weight, verbose,
               max_iter, decision_function_shape, break_ties, random_state)


def get_random_forest_classifier(n_estimators=100, criterion='gini', min_samples_split=2,
                                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                                 min_impurity_decrease=0.0, bootstrap=True, oob_score=False,
                                 verbose=0, warm_start=False, cp_alpha=0.0):

    return RandomForestClassifier(n_estimators, criterion, min_samples_split, min_samples_leaf,
                                  min_weight_fraction_leaf, max_features, min_impurity_decrease,
                                  bootstrap, oob_score, verbose, warm_start,
                                  cp_alpha)

def get_decision_tree_classifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2,
                                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                                 random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
                                 class_weight=None, ccp_alpha=0.0):

    return DecisionTreeClassifier(criterion, splitter, max_depth, min_samples_split, min_samples_leaf,
                                  min_weight_fraction_leaf, max_features, random_state, max_leaf_nodes,
                                  min_impurity_decrease, class_weight, ccp_alpha)


def compare_models(models, x_train, y_train, x_test, y_test):
    """
    expects a list of models as input
    """
    scores = []
    for model in models:
        model_name = type(model).__name__
        if model_name=='SVC' and model.kernel=='rbf': model_name+='RBF kernel'
        print("Fitting:", model_name)
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        macro_f1 = f1_score(y_test, prediction, average="macro")
        micro_f1 = f1_score(y_test, prediction, average="micro")
        weighted_f1 = f1_score(y_test, prediction, average="weighted")
        #print("y test:", y_test)
        #print("y pred:", prediction)
        recall = recall_score(y_test, prediction, average="macro")
        precision = precision_score(y_test, prediction, average="macro")
        
        score = model.score(x_test, y_test)
        scores.append((model_name, (f'{100*score:.2f}%'), str(macro_f1), str(micro_f1), str(weighted_f1), str(recall), str(precision)))

    scores_df = pd.DataFrame(scores, columns=['Model', 'Accuracy', "Macro F1", "Micro F1", "Weighted F1", "Recall", "Precision"])

    return scores_df



