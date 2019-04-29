from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
import numpy as np
import pandas as pd
import time


# Função para ler os dados do CSV
def load_antibot(name):
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
  ds = pd.read_csv(name+'.data', sep=',', header=None, names=cols)
  
  # Atributos em X
  X = ds.iloc[:, :-1]
  # Ultima linha (nossa classe) fica em y
  y = ds.iloc[:, -1]
  
  # Caso queira verificar algo nos atributos descomente a linha abaixo
  #print(X)
  
  return (X, y)
  
X, y = load_antibot("antibot")
print("Treinando o modelo.")
# 10-fold
skf = StratifiedKFold(10)
skf.split(X, y)
# Scalling
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
tempo_inicial = time.time()
# Best Params
model = SVC(C=3.0, degree=2, gamma='scale', kernel='poly')
model.fit(X, y)
scores_acc = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
scores_prec = cross_val_score(model, X, y, cv=skf, scoring='precision')
tempo_total = time.time() - tempo_inicial
print("taxa de acerto: %.2f%% +- %.2f%%, precisão: %.2f%% +- %.2f%% (%.3f (s))" % (scores_acc.mean() * 100, scores_acc.std() * 100, scores_prec.mean() * 100, scores_prec.std() * 100, tempo_total))

# Time to test what we have learned.
print("Testando o que foi aprendido.")
X_t, y_t = load_antibot("antibot2")
scaler.fit(X_t)
X_t = scaler.transform(X_t)
score = model.score(X_t, y_t)
print("Taxa de acerto: %.2f" % (score * 100))
predictions = model.predict(X_t)
print(classification_report(y_t, predictions))


