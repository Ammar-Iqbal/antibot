import pandas as pd
import numpy as np
import itertools
import math

# load critical values of F for the 0.05 significance level
critical_values = pd.read_csv('statistical/critical_values', sep=',', header=None)
critical_values.index += 1
critical_values.columns += 1

# load critical differences for the 0.05 significance level
critical_differences = pd.read_csv('statistical/critical_differences', sep=',', header=None, index_col=0)

df = pd.DataFrame()

def load_data(raw):
  data, index = {}, set([])
  for clf_name, ds_scores in raw.items():
    for ds_name, ds_score in ds_scores.items():
      data.setdefault(clf_name, []).append(ds_score[0])
      index.add(ds_name)
  global df
  df = pd.DataFrame(data, index=index)

def friedman():
  n, k = df.shape
  rank = df.rank(axis=1).sum()
  x = 12.0/(n*k*(k+1)) * np.sum((rank-n*(k+1)/2)**2)
  f = (n-1)*x/(n*(k-1)-x)
  critical_value = critical_values[k-1][(k-1)*(n-1)]
  return f > critical_value
  
def nemenyi():
  n, k = df.shape
  rank = df.rank(axis=1).mean()
  names = df.columns
  cd = critical_differences[k]['nemenyi']*math.sqrt(k*(k+1.0)/(6*n))
  res = pd.DataFrame(index=names, columns=names)
  for pair in list(itertools.combinations(zip(names, rank), 2)):
    res[pair[0][0]][pair[1][0]] = abs(pair[0][1] - pair[1][1]) >= cd
    res[pair[1][0]][pair[0][0]] = abs(pair[0][1] - pair[1][1]) >= cd
  return res
  