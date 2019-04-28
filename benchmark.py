from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import visualization as visual
import time

# Função para ler os dados do CSV
def load_antibot():
  # Listando nomes das colunas da base
  cols = [
    'qty_pick_bot',
    'reaction_pick_bot',
    'consistency_pick_bot',
    'avg_percent_life_bot',
    'reaction_heal_bot',
    'consistency_heal_bot',
    'qty_killed_bot',
    'avg_foodtime_bot',
    # preditor da classe (bot = 1, não bot = 0)
    'isbot']
    
  # Lendo todos os dados
  ds = pd.read_csv('antibot.data', sep=',', header=None, names=cols)
  
  # Atributos em X
  X = ds.iloc[:, :-1]
  # Ultima linha (nossa classe) fica em y
  y = ds.iloc[:, -1]
  
  # Caso queira verificar algo nos atributos descomente a linha abaixo
  #print(X)
  
  return ('antibot', X, y)

# Para facilitar caso queira analisar múltiplos datasets
datasets = []
datasets.append(load_antibot())

# Configurando os parâmetros dos classificadores
perceptron_params = {'perceptron__tol': [0.0001, 0.001, 0.01]}
mlp_params = {'mlp__hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)], 'mlp__activation': ['tanh', 'relu'], 'mlp__alpha': [0.0001, 0.05], 'mlp__learning_rate': ['constant','adaptive'],}
gaussian_nb_params = {}
decision_tree_params = {}
random_forest_params = {'randomforestclassifier__n_estimators': [1,3,5,7,9,11,13,15,17,19,21]}
logistic_regression_params = {'logisticregress__C': [0.001,0.01,0.1,1,10,100]}
linear_svc_params = {'linearsvc__C': np.linspace(0.5, 5., 10)}
knn_params = {'kneighborsclassifier__n_neighbors': [1,3,5,7,9,11,13,15,17,19,21]}
svc_linear_params = {'svc__kernel': ['linear'], 'svc__C': np.linspace(0.5, 5., 10)}
svc_poly_params = {'svc__kernel': ['poly'], 'svc__C': np.linspace(0.5, 5., 10), 'svc__degree': range(0, 5), 'svc__gamma': ['scale']}
svc_rbf_params = {'svc__kernel': ['rbf'], 'svc__C': np.linspace(0.5, 5., 10), 'svc__gamma': ['scale']}
svc_sigmoid_params = {'svc__kernel': ['sigmoid'], 'svc__C': np.linspace(0.5, 5., 10), 'svc__gamma': ['scale']}

# Criando nossos pipelines
pipes = []
pipes.append(('perceptron', perceptron_params, make_pipeline(SimpleImputer(), MinMaxScaler(), Perceptron())))
#pipes.append(('mlp', mlp_params, make_pipeline(SimpleImputer(), MinMaxScaler(), MLPClassifier())))
pipes.append(('naive_bayes', gaussian_nb_params, make_pipeline(SimpleImputer(), MinMaxScaler(), GaussianNB())))
pipes.append(('decision-tree', decision_tree_params, make_pipeline(SimpleImputer(), MinMaxScaler(), tree.DecisionTreeClassifier())))
pipes.append(('random-forest', random_forest_params, make_pipeline(SimpleImputer(), MinMaxScaler(), RandomForestClassifier())))
#pipes.append(('logistic-regression', logistic_regression_params, make_pipeline(SimpleImputer(), MinMaxScaler(), LogisticRegression())))
pipes.append(('linear-svc', linear_svc_params, make_pipeline(SimpleImputer(), MinMaxScaler(), LinearSVC())))
pipes.append(('knn', knn_params, make_pipeline(SimpleImputer(), MinMaxScaler(), KNeighborsClassifier())))
pipes.append(('svc-linear', svc_linear_params, make_pipeline(SimpleImputer(), MinMaxScaler(), SVC())))
pipes.append(('svc-poly', svc_poly_params, make_pipeline(SimpleImputer(), MinMaxScaler(), SVC())))
pipes.append(('svc-rbf', svc_rbf_params, make_pipeline(SimpleImputer(), MinMaxScaler(), SVC())))
pipes.append(('svc-sigmoid', svc_sigmoid_params, make_pipeline(SimpleImputer(), MinMaxScaler(), SVC())))

data = {}
bestparams = []
for ds_name, X, y in datasets:
  # Separação da base entre treino e teste com 10 pastas
  skf = StratifiedKFold(10)
  skf.split(X, y)
  # Plota a arvore de decisão pro modelo
  clf = tree.DecisionTreeClassifier()
  clf.fit(X,y)
  tree.export_graphviz(clf, out_file='tree.dot')
  print("\nTestando a base " + ds_name + ":\n[Método] = Média Acc. +- std, Média Roc +- std. (Tempo)")
  for clf_name, params, pipe in pipes:
    # Salvando o tempo para benchmark 
    tempo_inicial = time.time()
    # Procurando os melhores parâmetros para cada algoritmo
    best_attemp = GridSearchCV(pipe, params, cv=skf).fit(X, y)
    # Extraindo os parametros da melhor tentativa
    best_params = best_attemp.best_params_
    bestparams.append(best_params)
    pipe.set_params(**best_params)
    # Computando os scores
    scores_acc = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy')
    scores_roc = cross_val_score(pipe, X, y, cv=skf, scoring='roc_auc')
    data.setdefault(clf_name, {})[ds_name] = ((scores_acc.mean(), scores_acc.std()))
    # Computando tempo de execução
    tempo_total = time.time() - tempo_inicial
    # Retornando resultados
    print("[%s] = taxa de acerto: %.2f%% +- %.2f%%, roc_auc: %.2f%% +- %.2f%% (%.3f (s))" % (clf_name, scores_acc.mean() * 100, scores_acc.std() * 100, scores_roc.mean() * 100, scores_roc.std() * 100, tempo_total))
np.savetxt("best_params", bestparams, fmt='%s', delimiter=",")

visual.load_data(data)
visual.to_csv('benchmark.csv')
visual.to_chart('benchmark.png')