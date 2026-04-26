import pandas as pd
from seaborn import cm
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

def run_eval_matrix(configs, X_test, y_test):
  """
  Takes a list of model configs, trains and evaluates them on the provided test
  set and returns a DataFrame of results.
  """
  eval_results = []
  for config in configs:
    method_name = config['Sampling_Method']
    X_tr = config['X_train']
    y_tr = config['y_train']

    for model_name, model in config['models'].items():
      model.fit(X_tr, y_tr)
      y_pred = model.predict(X_test)
      f1 = f1_score(y_test, y_pred, average='macro')
      precision = precision_score(y_test, y_pred, average='macro')
      recall = recall_score(y_test, y_pred, average='macro')
      cm = confusion_matrix(y_test, y_pred)

      eval_results.append(
        {
          "Method": method_name,
          "Algorithm": model_name,
          "F1_Score": round(f1, 4),
          "Precision": round(precision, 4),
          "Recall": round(recall, 4),
          "Confusion Matrix": cm
        }
      )

  df = pd.DataFrame(eval_results)
  df.sort_values(by=['Method', 'Algorithm'], inplace=True)

  return df
