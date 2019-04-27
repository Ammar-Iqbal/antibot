from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import Imputer, MinMaxScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import comparison as comp
import visualization as visual

#declaring the function to load the data from CSV
def load_antibot():
  # attribute information
  cols = [
    # preditor da classe (bot = 1, não bot = 0)
    'qty_pick_bot',
    'reaction_pick_bot',
    'consistency_pick_bot',
    'avg_percent_life_bot',
    'reaction_heal_bot',
    'consistency_heal_bot',
    'qty_killed_bot',
    'avg_foodtime_bot',
    'isbot']
    
  # load dataset
  ds = pd.read_table('antibot.data', sep=',', header=None, names=cols)
  
  X = ds.iloc[:, :-1]
  y = ds.iloc[:, -1]
  
  return ('antibot', X, y)

# load datasets
datasets = []
datasets.append(load_antibot())


# configure classifier parameters (best)
linear_svc_params = {'linearsvc__C': np.linspace(0.5, 5., 10)}
knn_params = {'kneighborsclassifier__n_neighbors': [1,3,5,7,9,11,13,15,17,19,21]}
svc_linear_params = {'svc__kernel': ['linear'], 'svc__C': np.linspace(0.5, 5., 10)}
svc_poly_params = {'svc__kernel': ['poly'], 'svc__C': np.linspace(0.5, 5., 10), 'svc__degree': range(0, 5)}
svc_rbf_params = {'svc__kernel': ['rbf'], 'svc__C': np.linspace(0.5, 5., 10)}
svc_sigmoid_params = {'svc__kernel': ['sigmoid'], 'svc__C': np.linspace(0.5, 5., 10)}

# configure classifier parameters (worst but with best K)
#linear_svc_params = {}
#knn_params = {'kneighborsclassifier__n_neighbors': [1,3,5,7,9,11,13,15,17,19,21]}
#svc_linear_params = {'svc__kernel': ['linear']}
#svc_poly_params = {'svc__kernel': ['poly']}
#svc_rbf_params = {'svc__kernel': ['rbf']}
#svc_sigmoid_params = {'svc__kernel': ['sigmoid']}

pipes = []
pipes.append(('linear-svc', linear_svc_params, make_pipeline(Imputer(), MinMaxScaler(), LinearSVC())))
pipes.append(('knn', knn_params, make_pipeline(Imputer(), MinMaxScaler(), KNeighborsClassifier())))
pipes.append(('svc-linear', svc_linear_params, make_pipeline(Imputer(), MinMaxScaler(), SVC())))
pipes.append(('svc-poly', svc_poly_params, make_pipeline(Imputer(), MinMaxScaler(), SVC())))
pipes.append(('svc-rbf', svc_rbf_params, make_pipeline(Imputer(), MinMaxScaler(), SVC())))
pipes.append(('svc-sigmoid', svc_sigmoid_params, make_pipeline(Imputer(), MinMaxScaler(), SVC())))

data = {}
for ds_name, X, y in datasets:
  skf = StratifiedKFold(10)
  skf.split(X, y)
  for clf_name, params, pipe in pipes:
    best_params = GridSearchCV(pipe, params, cv=skf).fit(X, y).best_params_
    pipe.set_params(**best_params)
    scores = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy')
    data.setdefault(clf_name, {})[ds_name] = ((scores.mean(), scores.std()))
    print(ds_name)#, clf_name, scores.mean(), scores.std())

comp.load_data(data)
if (comp.friedman()):
  print('H0 was rejected')
  print(comp.nemenyi())
else:
  print('H0 was not rejected')
  
visual.load_data(data)
visual.to_csv('benchmark.csv')
visual.to_chart('benchmark.png')