import numpy as np
import os
import pandas as pd

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# use sklearn models as baseline
# calling these without arguments will result in default settings

def get_knn_regressor(n_neighbors=5, weights="uniform", algorithm="auto",
                  leaf_size=30, p=2, metric_params=None, n_jobs=None):

    return KNeighborsRegressor(n_neighbors, weights, algorithm, leaf_size, p, metric_params, n_jobs)


def get_knn_classifier(n_neighbors=5, weights="uniform", algorithm="auto",
                  leaf_size=30, p=2, metric_params=None, n_jobs=None):
    
    return KNeighborsClassifier(n_neighbors, weights, algorithm, leaf_size, p, metric_params, n_jobs)


def get_svm_classifier(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False,
                       tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1,
                       decision_function_shape='ovr', break_ties=False, random_state=None):

    return SVC(C, kernel, degree, gamma, coef0, shrinking, probability, tol, cache_size, class_weight, verbose,
               max_iter, decision_function_shape, break_ties, random_state)


def get_random_forest_classifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2,
                                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                                 max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False,
                                 n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None,
                                 ccp_alpha=0.0, max_samples=None):

    return RandomForestClassifier(n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf,
                                  min_weight_fraction_leaf, max_features, max_leaf_nodes, min_impurity_decrease,
                                  bootstrap, oob_score, n_jobs, random_state, verbose, warm_start, class_weight,
                                  ccp_alpha=, max_samples)

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
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)
        model_name = type(model).__name__
        scores.append((model_name, (f'{100*score:.2f}%')))

    scores_df = pd.DataFrame(scores, columns=['Model', 'Score'])

    return scores_df



