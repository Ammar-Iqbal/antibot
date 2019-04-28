from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import numpy as np
import pandas as pd
import comparison as comp
import visualization as visual
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


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
linear_svc_params = {'linearsvc__C': np.linspace(0.5, 5., 10)}
knn_params = {'kneighborsclassifier__n_neighbors': [1,3,5,7,9,11,13,15,17,19,21]}
svc_linear_params = {'svc__kernel': ['linear'], 'svc__C': np.linspace(0.5, 5., 10)}
svc_poly_params = {'svc__kernel': ['poly'], 'svc__C': np.linspace(0.5, 5., 10), 'svc__degree': range(0, 5), 'svc__gamma': ['scale']}
svc_rbf_params = {'svc__kernel': ['rbf'], 'svc__C': np.linspace(0.5, 5., 10), 'svc__gamma': ['scale']}
svc_sigmoid_params = {'svc__kernel': ['sigmoid'], 'svc__C': np.linspace(0.5, 5., 10), 'svc__gamma': ['scale']}

# Criando nossos pipelines
pipes = []
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
    intermed = GridSearchCV(pipe, params, cv=skf)
    # Calculando o fit da melhor tentativa
    best_attemp = intermed.fit(X, y)
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
    # contém erro por enquanto...
    #plot_learning_curve(pipe, "Curva de aprendizado", X, y, ylim=(0.7, 1.01), cv=skf)
    #plt.show()
    print("[%s] = taxa de acerto: %.3f +- %.3f, roc_auc: %.3f +- %.3f (%.3f (s))" % (clf_name, scores_acc.mean(), scores_acc.std(), scores_roc.mean(), scores_roc.std(), tempo_total))
np.savetxt("best_params", bestparams, fmt='%s', delimiter=",")
comp.load_data(data)

visual.load_data(data)
visual.to_csv('benchmark.csv')
visual.to_chart('benchmark.png')