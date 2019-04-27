import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

df = pd.DataFrame()

def load_data(raw):
  data, index = {}, set([])
  for clf_name, ds_scores in raw.items():
    for ds_name, ds_score in ds_scores.items():
      data.setdefault(clf_name + '_mean', []).append(ds_score[0])
      data.setdefault(clf_name + '_std', []).append(ds_score[1])
      index.add(ds_name)
  global df
  df = pd.DataFrame(data, index=index)

def to_chart(filepath, errorbar = False):
  xticks = df.index.values
  columns = [col.replace('_mean', '') for col in list(df) if col.endswith('_mean')]
  x = range(0, len(xticks))
  for column in columns:
    if not errorbar:
      plt.plot(x, df[column + '_mean'], label=column)
    else:
      plt.errorbar(x, df[column + '_mean'], yerr=df[column + '_std'], label=column)
  plt.xticks(x, xticks, rotation='vertical')
  plt.legend(bbox_to_anchor=(1, 1))
  plt.savefig(filepath, bbox_inches='tight')
  plt.clf()
  
def to_csv(filepath):
  df.to_csv(filepath)