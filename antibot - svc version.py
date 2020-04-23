from datasets.generator import AntibotDataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate

from sklearn.svm import SVC
from sklearn.metrics import classification_report

import pandas as pd
import numpy as np

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


def load_antibot_train():
    ds = pd.read_csv('antibot.train.data', sep=',',
                     header=None, names=cols, dtype=float)

    X = ds.iloc[:, :-1]
    y = ds.iloc[:, -1]

    return (X, y)


def load_antibot_test():
    ds = pd.read_csv('antibot.test.data', sep=',',
                     header=None, names=cols, dtype=float)

    X = ds.iloc[:, :-1]
    y = ds.iloc[:, -1]

    return (X, y)


AntibotDataset(1000, 0.2, 255).generate().export_csv("antibot.train.data")
AntibotDataset(1000, 0.1, 510).generate().export_csv("antibot.test.data")

X, y = load_antibot_train()
X_t, y_t = load_antibot_test()

# Stratified 10-Fold
skf = StratifiedKFold(10)
skf.split(X, y)

# Estimator
estimator = SVC(C=3.0, degree=2, gamma='scale', kernel='poly')

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'f1': 'f1',
    'recall': 'recall',
    'roc_auc': 'roc_auc'
}
scores = cross_validate(make_pipeline(MinMaxScaler(), estimator), X, y,
                        cv=skf, scoring=scoring, return_train_score=False)

print("SVC(C=3.0, degree=2, gamma='scale', kernel='poly')")
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

# Scale training dataset
scaler = MinMaxScaler()
scaler.fit(X)

# Fit estimator with the entire training dataset
estimator.fit(scaler.transform(X), y)

# Predict with unseen dataset
predictions = estimator.predict(scaler.transform(X_t))

# Print report
print(classification_report(y_t, predictions))

print("Indices of support vectors")
print(estimator.support_)
print()

print("Number of support vectors for each class")
print(estimator.n_support_)
print()
