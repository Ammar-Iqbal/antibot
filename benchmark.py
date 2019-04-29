from datasets.generator import AntibotDataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd


def load_antibot():
    cols = [
        'collected_items',
        'avg_time_to_collect_item',
        'delta_time_to_collect_item',
        'heal_threshold',
        'avg_time_to_start_healing',
        'delta_time_to_start_healing',
        'killed_enemies',
        'hungry_time',
        'class'
    ]

    ds = pd.read_csv('antibot.data', sep=',',
                     header=None, names=cols, dtype=float)

    X = ds.iloc[:, :-1]
    y = ds.iloc[:, -1]

    return ('antibot', X, y)


AntibotDataset(1000, 0.2, 255).generate().export_csv("antibot.data")

datasets = []
datasets.append(load_antibot())

decisiontreeclassifier_params = {
    'decisiontreeclassifier__criterion': ['gini', 'entropy'],
    'decisiontreeclassifier__splitter': ['best', 'random'],
    'decisiontreeclassifier__min_samples_split': [2, 4, 6, 8, 10],
    'decisiontreeclassifier__min_samples_leaf': [1, 3, 5, 7, 9],
}
kneighborsclassifier_params = {
    'kneighborsclassifier__n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
    'kneighborsclassifier__weights': ['uniform', 'distance'],
    'kneighborsclassifier__algorithm': ['auto', 'brute', 'kd_tree', 'ball_tree'],
}
linearsvc_params = {
    'linearsvc__C': np.linspace(0.5, 5., 10),
}
svc_linear_params = {
    'svc__kernel': ['linear'],
    'svc__C': np.linspace(0.5, 5., 10),
}
svc_poly_params = {
    'svc__kernel': ['poly'],
    'svc__C': np.linspace(0.5, 5., 10),
    'svc__degree': range(0, 10),
    'svc__gamma': ['scale'],
}
svc_rbf_params = {
    'svc__kernel': ['rbf'],
    'svc__C': np.linspace(0.5, 5., 10),
    'svc__gamma': ['scale'],
}

pipes = []
pipes.append(('DecisionTreeClassifier', decisiontreeclassifier_params,
              make_pipeline(MinMaxScaler(), DecisionTreeClassifier())))
pipes.append(('KNeighborsClassifier', kneighborsclassifier_params,
              make_pipeline(MinMaxScaler(), KNeighborsClassifier())))
pipes.append(('LinearSVC', linearsvc_params,
              make_pipeline(MinMaxScaler(), LinearSVC())))
pipes.append(('SVC_linear', svc_linear_params,
              make_pipeline(MinMaxScaler(), SVC())))
pipes.append(('SVC_poly', svc_poly_params,
              make_pipeline(MinMaxScaler(), SVC())))
pipes.append(('SVC_rbf', svc_rbf_params,
              make_pipeline(MinMaxScaler(), SVC())))

data = {}
bestparams = []
for ds_name, X, y in datasets:
    # Separação da base entre treino e teste com 10 pastas
    skf = StratifiedKFold(10)
    skf.split(X, y)

    for clf_name, params, pipe in pipes:
        best_attempt = GridSearchCV(pipe, params, cv=skf).fit(X, y)

        best_params = best_attempt.best_params_
        bestparams.append(best_params)
        pipe.set_params(**best_params)

        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'f1': 'f1',
            'recall': 'recall',
            'roc_auc': 'roc_auc'
        }
        scores = cross_validate(
            pipe, X, y, cv=skf, scoring=scoring, return_train_score=False)

        print(clf_name)
        print("    accuracy: %.3f +/- %.3f" %
              (scores['test_accuracy'].mean(), scores['test_accuracy'].std()))
        print("    precision: %.3f +/- %.3f" %
              (scores['test_precision'].mean(), scores['test_precision'].std()))
        print("    f1: %.3f +/- %.3f" %
              (scores['test_f1'].mean(), scores['test_f1'].std()))
        print("    recall: %.3f +/- %.3f" %
              (scores['test_recall'].mean(), scores['test_recall'].std()))
        print("    roc_auc: %.3f +/- %.3f" %
              (scores['test_roc_auc'].mean(), scores['test_roc_auc'].std()))
        print()

        data.setdefault(clf_name, {})[ds_name] = scores

np.savetxt("best_params", bestparams, fmt='%s', delimiter=",")
