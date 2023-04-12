import re
import pandas as pd
import time
import json
import pickle
from glob import glob
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def load_model(modle_path:str):

      with open(modle_path, 'rb') as f:
            model = pickle.load(f)
      f.close()

      return model


def save_cluster_model(model_name:str, model):
      with open(f'./user_cluter_models/{model_name}_model.pkl', 'wb') as f:
            pickle.dump(model, f)


def is_only_contain_letters(string:str):

      res = re.findall(re.compile(r'[A-Za-z]', re.S), string)
      
      if len(res):
            return True
      else:
            return False


def Levenshtein_Distance_Similarity(s1:str, s2:str):

      len_s1 = len(s1)
      len_s2 = len(s2)

      dp = [[0 for _ in range(len_s2 + 1)] for _ in range(len_s1 + 1)]
      for i in range(len_s1 + 1):
            for j in range(len_s2 + 1):
                  if i == 0:
                        dp[i][j] = j
                  elif j == 0:
                        dp[i][j] = j
                  elif s1[i-1] == s2[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                  else:
                        dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
      
      result = 1 - dp[len_s1][len_s2] / max(len_s1, len_s2)
      
      return result 


def get_user_path_matrix(data:pd.DataFrame, users:list):
      
      for user_name in users:

            path_dict={}

            ori_path_list = list(data[data['User'] == user_name]['Path'].unique())
            

            path_list = []
            for path in ori_path_list: 
                  cop = re.compile(r'[^a-z^A-Z^0-9^/^]')
                  re_path = cop.sub('', str(path)).lower()
                  path_list.append(re_path)

            new_path_list = list(set(path_list))
            new_path_list.sort(key = path_list.index)

            json_path_data = json.dumps(new_path_list, ensure_ascii=False, separators=(',',': '), indent=2)

            with open(f'./user_path/{user_name}.json', 'w', encoding='utf-8') as f:
                  f.writelines(json_path_data)
            f.close()


            for i in range(len(new_path_list)):
                  dist_list=[]

                  temp = [0.0] * i + [1.0]
                  
                  if i < len(new_path_list) - 1:
                        for j in range(i+1,len(new_path_list)):

                              dist_list.append(Levenshtein_Distance_Similarity(new_path_list[i], new_path_list[j]))
                        
                        total_dist = temp + dist_list
                        path_dict[new_path_list[i]] = total_dist
                  else:

                        total_dist = temp + dist_list
                        path_dict[new_path_list[i]] = total_dist
            
            path_df = pd.DataFrame(path_dict)
            

            for m in range(1, path_df.shape[1]):
                  for n in range(m):
                        path_df.iloc[n,m] = path_df.iloc[m,n]

            path_df.to_csv(f'./user_dist_matrix/{user_name}.csv', index=False)


def cluster_train_path():

      path_matrix_files = glob('./user_dist_matrix/*.csv')

      for file_ in path_matrix_files:
           
            user_name = file_.rsplit('/')[-1].split('.')[0]

            path_df = pd.read_csv(file_)

            path_df_ = PCA(n_components=0.98, random_state=0).fit_transform(path_df)
            new_path_df = StandardScaler().fit_transform(path_df_)

            input_component = new_path_df.shape[1] 
            dist = []

            k = 2

            flag =False
            while flag == False:

                  clu = KMeans(n_clusters=k, n_jobs=-1, random_state=0).fit(new_path_df)
                  
                  model_name = user_name + '_' + str(input_component)
                  save_cluster_model(model_name=model_name, model=clu)
                  path_df['label'] = clu.labels_


                  dist = []
                  for i in path_df.label.unique():

                        col = path_df.columns[path_df[path_df['label'] == i].index]

                        if len(col) != 1:

                              for n in range(len(col)):
                                    for m in range(n+1, len(col)):
                                          dist.append(Levenshtein_Distance_Similarity(col[n],col[m]))
                                          
                        else:
                              dist.append(1)

                  if all(x > 0.8 for x in dist):

                        flag = True

                  k = k + 1


            clustering_path_dict = {}
            clustering_path_dict_hash = {}

            for s in range(len(list(path_df.columns)[:-1])):
                  clustering_path_dict[path_df.columns[s]] = user_name + str(clu.labels_[s])
            
            for s in range(len(list(path_df.columns)[:-1])):
                  clustering_path_dict_hash[user_name +  str(clu.labels_[s])] = hash(user_name +  str(clu.labels_[s]))

            json_data = json.dumps(clustering_path_dict, ensure_ascii=False, separators=(',',': '), indent=2)

            json_data_hash = json.dumps(clustering_path_dict_hash, ensure_ascii=False, separators=(',',': '), indent=2)

            with open(f'./path_class/{user_name}.json', 'w', encoding='utf-8') as f:
                  f.writelines(json_data)
            f.close()

            with open(f'./path_class_hash/{user_name}.json', 'w', encoding='utf-8') as f:
                  f.writelines(json_data_hash)
            f.close()


def cluster_test_path(data:pd.DataFrame, ori_path_list:list):
      
      model_file = glob('./user_cluter_models/*.pkl')

      total_df_list = []
      for model in model_file:
            

            clu_model = load_model(model)

            input_component = int(model.rsplit('/', 1)[-1].split('.')[0].split('_')[1])

            user_name = model.rsplit('/', 1)[-1].split('_')[0]

            with open(f'./user_path/{user_name}.json', 'r') as f:
                  model_path_data = json.load(f)
            f.close()

            path_list = []
            for path in ori_path_list: 
                  cop = re.compile(r'[^a-z^A-Z^0-9^/^]')
                  re_path = cop.sub('', str(path)).lower()
                  path_list.append(re_path)


            test_path_dict = {}
            for i in range(len(model_path_data)):

                  dist_list=[]

                  for j in range(len(path_list)):

                        dist_list.append(Levenshtein_Distance_Similarity(model_path_data[i], path_list[j]))
                  
                  test_path_dict[model_path_data[i]] = dist_list

            final_path_df = pd.DataFrame(test_path_dict)

            path_df_ = PCA(n_components=input_component, random_state=0).fit_transform(final_path_df)
            final_path_df = StandardScaler().fit_transform(path_df_)

            labels = [user_name +  str(x) for x in list(clu_model.predict(final_path_df))]
            
            temp_df = data[data['User'] == user_name]
            temp_df['PathType'] = labels

            total_df_list.append(temp_df)

            result_df = pd.concat(total_df_list)
            result_df = result_df.sort_values(by='JobID', ascending=True)
            
            return result_df
      

def cluster_train_jobname(data:pd.DataFrame, ori_jobname_list:list):
      
      # Clustering Jobname
      retained_jobname_list = []

      for jobname in ori_jobname_list:
            jobname = jobname.rsplit('/', 1)[-1]
            if is_only_contain_letters(str(jobname)):
                  cop = re.compile(r'[^a-z^A-Z^\-^_^/^.^+^=^(^)^,^ ^]')
                  jobname = cop.sub('', str(jobname)).lower()
                  retained_jobname_list.append(jobname)
            else:
                  if str(jobname).isdigit():
                        retained_jobname_list.append(str(len(jobname)))
                  else:
                        cop = re.compile(r'[^\-^_^/^.^+^=^(^)^,^ ^]')
                        jobname = cop.sub('', str(jobname)).lower()
                        retained_jobname_list.append(jobname)

      # Adding Clustrred Data
      data['JobType'] = [hash(x) for x in  retained_jobname_list]

      clustering_jobname_dict_hash = {}
      for s in range(len(retained_jobname_list)):
            clustering_jobname_dict_hash[retained_jobname_list[s]] = hash(retained_jobname_list[s])

      json_data_hash = json.dumps(clustering_jobname_dict_hash, ensure_ascii=False, separators=(',',': '), indent=2)

      with open(f'./jobname_hash.json', 'w', encoding='utf-8') as f:
            f.writelines(json_data_hash)
      f.close()

      return data


def cluter_test_jobname(data:pd.DataFrame, ori_jobname_list:list):

      with open(f'./jobname_hash.json', 'r') as f:
            jobname_data = json.load(f)
      f.close()

       # Clustering Jobname
      retained_jobname_list = []

      for jobname in ori_jobname_list:
            jobname = jobname.rsplit('/', 1)[-1]
            if is_only_contain_letters(str(jobname)):
                  cop = re.compile(r'[^a-z^A-Z^\-^_^/^.^+^=^(^)^,^ ^]')
                  jobname = cop.sub('', str(jobname)).lower()
                  retained_jobname_list.append(jobname)
            else:
                  if str(jobname).isdigit():
                        retained_jobname_list.append(str(len(jobname)))
                  else:
                        cop = re.compile(r'[^\-^_^/^.^+^=^(^)^,^ ^]')
                        jobname = cop.sub('', str(jobname)).lower()
                        retained_jobname_list.append(jobname)
      
      # Adding Clustrred Data
      jobname_hash_list = []
      for x in retained_jobname_list:
            if x in list(jobname_data.keys()):
                  jobname_hash_list.append(jobname_data[x])
            else:
                  jobname_hash_list.append(hash(x))
                  
      data['JobType'] = jobname_hash_list

      return data


def get_timestamp_dataframe(data:pd.DataFrame, ori_time_list:list):
      
      time_stamp_list = []
      for time_str in ori_time_list:
            
            time_format =  '%Y-%m-%dT%H:%M:%S'
            time_tuple = time.strptime(time_str, time_format)
            time_stamp = int(time.mktime(time_tuple))
            time_stamp_list.append(time_stamp)
      
      data['Submit'] = time_stamp_list
      
      return data

