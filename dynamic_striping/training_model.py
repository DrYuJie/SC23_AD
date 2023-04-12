import pandas as pd
import numpy as np
import re
import json
from glob import glob

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

from utils.data_preprocessing import is_only_contain_letters, cluster_train_jobname, cluster_test_jobname, get_timestampe_dataframe, get_user_path_matrix, cluster_train_path, cluster_test_path
import pickle



def bin_classify(model_name:str, clf, X_train, X_test, y_train, transform=True, params=None, score=None):

      
      grid_search = model_selection.GridSearchCV(estimator=clf, param_grid=params, cv=3, scoring=score, n_jobs=-1)
      
      if transform:
            X_train_tr = StandardScaler().fit_transform(X_train)
            X_test_tr = StandardScaler().fit_transform(X_test)
            grid_search.fit(X_train_tr, y_train)
            y_pred = grid_search.predict(X_test_tr)
            if hasattr(grid_search,'predict_proba'):
                  y_score = grid_search.predict_proba(X_test_tr)[:,1]
            elif hasattr(grid_search, 'decision_function'):
                  y_score = grid_search.decision_function(X_test_tr)
            else:
                  y_score = y_pred
      else:
            grid_search.fit(X_train, y_train)
            y_pred = grid_search.predict(X_test)
            if hasattr(grid_search,'predict_proba'):
                  y_score = grid_search.predict_proba(X_test)[:, 1]
            elif hasattr(grid_search, 'decision_function'):
                  y_score = grid_search.decision_function(X_test)
            else:
                  y_score = y_pred
      
      
      predictions = {'y_pred': y_pred, 'y_score': y_score}
      df_predictions = pd.DataFrame.from_dict(predictions)
      return grid_search.best_estimator_, df_predictions


def bin_class_metrics(model_name:str, y_test, y_pred, print_out=True):

      Matrix = metrics.confusion_matrix(y_test,y_pred)

      N = len(y_test[y_test==0])
      specificity = Matrix[0][0] / N

      P = len(y_test[y_test==1])
      sensitivity = Matrix[1][1] / P

      TOTAL = len(list(y_test))

      S_score1 = 2 * specificity * sensitivity / (specificity + sensitivity)
      S_score2 = 5 * (specificity * sensitivity) / (4 * specificity + sensitivity)

      bin_class_metrics ={
            'Accuracy': metrics.accuracy_score(y_test, y_pred),
            'Special': specificity,
            'Sensitivity': sensitivity,
            'S_Score1': S_score1,
            'S_Score2':S_score2,
            'N' : N,
            'P': P,
            'Total Number': TOTAL,
      }

      df_metrics = pd.DataFrame.from_dict(bin_class_metrics, orient='index')
      df_metrics.columns = [model_name]

      if print_out:
            print('-' * 120)
            print(model, '\n')
            print('Confusion  Matrix:')
            print(metrics.confusion_matrix(y_test, y_pred))
            print('\nClassfication Report:')
            print(metrics.classification_report(y_test, y_pred))
            print('\nMetrics:')
            print(df_metrics)
      
      return df_metrics


def oversample(input_data:pd.DataFrame, input_number:int):

      over_sample = input_data[input_data['Label'] ==1].sample(n=over_sample_number, replace =True, random_state=12)

      final_train_data = pd.concat([over_sample, input_data], axis=0)

      return final_train_data


def save_model(model_name:str, clf):
      with open(f'./{model_name}_model.pkl', 'wb') as f:
            pickle.dump(clf, f)


if __name__ == "__main__":

      total_data_df = pd.read_csv('./total_data.csv')
      
      # Data Preprocessing
      total_data_df = get_timestampe_dataframe(data=total_data_df, ori_time_list=list(total_data_df['Submit']))
      total_data_df = total_data_df.sort_values(by='JobID', ascending=True)

      features = ['UID', 'ReqCPUS', 'Submit', 'JobType', 'NNodes', 'PathType']
      
      # Splitting Data
      train_data = total_data_df.iloc[:int(total_data_df.shape[0]*0.7)]
      get_user_path_matrix(data=train_data, users=list(train_data['User'].unique()))
      cluster_train_path()
      

      train_data_list = []
      for user_name in list(train_data['User'].unique()):

            with open(f'./path_class/{user_name}.csv', 'r') as f:
                  model_path_data = json.load(f)
            f.close()

            with open(f'./path_class_hash/{user_name}.csv', 'r') as f:
                  model_path_data_hash = json.load(f)
            f.close()

            new_train_data = train_data[train_data['User'] == user_name]
            new_train_data['PathType'] = [model_path_data_hash[model_path_data[x]] for x in new_train_data['Path']]
            train_data_list.append(new_train_data)

      train_data = pd.concat(train_data_list)
      train_data = cluster_train_jobname(data=train_data, ori_jobname_list=list(train_data['JobName']))


      test_data = total_data_df.iloc[int(total_data_df.shape[0]*0.7):]
      test_data = cluster_test_path(data=test_data, ori_path_list=list(test_data['Path']))
      test_data = cluster_test_jobname(data=test_data, ori_jobname_list=list(test_data['JobName']))
      
      over_sample_number = train_data[train_data['Label'] == 0].shape[0] - train_data[train_data['Label'] == 1].shape[0]

      if over_sample_number > 0:
            train_data = oversample(input_data=train_data, input_number=over_sample_number)
      
      
      
      X_train = train_data[features]
      y_train = train_data['Label']

      X_test = test_data[features]
      y_test = test_data['Label']

      
      # 'Decision Tree'
      model = 'DT'
      clf_dtc = DecisionTreeClassifier(random_state=12)
      gs_params = {'criterion':['gini', 'entropy'], 'max_depth':[3, 4, 5, 6, 7]}
      gs_score = 'accuracy'

      clf_dtc, pred_dtc = bin_classify(model_name=model, clf=clf_dtc, X_train=X_train, X_test=X_test, y_train=y_train, transform=False, params=gs_params, score=gs_score)
      print('\n Best Parameters:\n', clf_dtc)

      metrics_dtc = bin_class_metrics(model_name=model, y_test=y_test, y_pred=pred_dtc.y_pred)
      print(metrics_dtc)

      save_model(model_name=model, clf=clf_dtc)


      # Random Forest
      model = 'RF'
      clf_rfc = RandomForestClassifier(random_state=12, n_jobs=-1)
      gs_params = {'max_depth':[4, 5, 6, 7, 8, 9, 10], 'criterion':['gini', 'entropy'], 'n_estimators':[50, 60, 70, 80, 90, 100, 110]}
      gs_score = 'accuracy'

      clf_rfc, pred_rfc = bin_classify(model_name=model, clf=clf_rfc, X_train=X_train, X_test=X_test, y_train=y_train, transform=False, params=gs_params, score=gs_score)

      print('\nBest Parameters:\n', clf_rfc)

      metrics_rfc = bin_class_metrics(model_name=model, y_test=y_test, y_pred=pred_rfc.y_pred)
      print(metrics_rfc)

      save_model(model_name=model, clf=clf_rfc)


      # Extra Trees
      model = 'ET'
      clf_etc = ExtraTreesClassifier(random_state=12, n_jobs=-1)
      gs_params = {'max_depth':[4, 5, 6, 7, 8, 9, 10], 'criterion':['gini', 'entropy'], 'n_estimators':[50, 60, 70, 80, 90, 100, 110]}
      gs_score = 'accuracy'

      clf_etc, pred_etc = bin_classify(model_name=model, clf=clf_rfc, X_train=X_train, X_test=X_test, y_train=y_train, transform=False, params=gs_params, score=gs_score)

      print('\nBest Parameters:\n', clf_etc)

      metrics_etc = bin_class_metrics(model_name=model, y_test=y_test, y_pred=pred_etc.y_pred)
      print(metrics_etc)

      save_model(model_name=model, clf=clf_etc)