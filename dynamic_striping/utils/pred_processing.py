import re
import time
import json
import pandas as pd
from data_preprocessing import Levenshtein_Distance_Similarity, load_model
from glob import glob
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def get_absolute_jobname(ori_jobname:str):

      absolute_jobname = ori_jobname.rsplit('/', 1)[-1]

      return absolute_jobname


def is_only_contain_letters(string:str):

      res = re.findall(re.compile(r'[A-Za-z]', re.S), string)
      
      if len(res):
            return True
      else:
            return False


def cluster_jobname(ori_jobname:str):
      
      with open(f'./jobname_hash.json', 'r') as f:
            jobname_data = json.load(f)
      f.close()

      # Clustering Jobname
      if is_only_contain_letters(str(ori_jobname)):
            cop = re.compile(r'[^a-z^A-Z^\-^_^/^.^+^=^(^)^,^ ^]')
            jobname = cop.sub('', str(ori_jobname)).lower()
      else:
            if str(jobname).isdigit():
                  jobname = str(len(ori_jobname))
            else:
                  cop = re.compile(r'[^\-^_^/^.^+^=^(^)^,^ ^]')
                  jobname = cop.sub('', str(ori_jobname)).lower()

      if jobname in list(jobname_data.keys()):
            jobname = jobname_data[jobname]
      else:
            jobname = hash(jobname)

      return jobname


def get_timestamp(time_str:str):

      time_format =  '%Y-%m-%dT%H:%M:%S'
      time_tuple = time.strptime(time_str, time_format)
      time_stamp = int(time.mktime(time_tuple))
      return time_stamp


def cluster_path(user_name:str, path:str):

      cluster_model_files = glob(f'./user_cluter_models/*.pkl')
      for file_ in cluster_model_files:
            if user_name in file_:
                  model_dir = file_
      
      clu_model = load_model(modle_path=model_dir)

      input_component = int(model_dir.rsplit('/', 1)[-1].split('.')[0].split('_')[1])

      with open(f'./user_path/{user_name}.json', 'r') as f:
            model_path_data = json.load(f)
      f.close()

      cop = re.compile(r'[^a-z^A-Z^0-9^/^]')
      re_path = cop.sub('', str(path)).lower()


      test_path_dict = {}
      for i in range(len(model_path_data)):
            dist_list=[]
            dist_list.append(Levenshtein_Distance_Similarity(model_path_data[i], re_path))
            test_path_dict[model_path_data[i]] = dist_list

      final_path_df = pd.DataFrame(test_path_dict)

      path_df_ = PCA(n_components=input_component, random_state=0).fit_transform(final_path_df)

      final_path_df = StandardScaler().fit_transform(path_df_)

      path_class = user_name +  str(clu_model.predict(final_path_df))

      with open(f'./path_class_hash/{user_name}.csv', 'r') as f:
            model_path_data_hash = json.load(f)
      f.close()
      
      result = model_path_data_hash[path_class]

      return result
