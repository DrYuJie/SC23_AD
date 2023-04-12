import os
import time
import sys
import pickle
import numpy as np
from utils.pred_processing import is_only_contain_letters, cluster_jobname, get_absolute_jobname, get_timestamp, cluster_path

def is_null(file_name:str):
    size = os.path.getsize(file_name)
    
    if size == 0:
        return True
    else:
        return False

def import_model(model_file:str):
    
    with open(f'{model_file}', 'rb') as f:
        model = pickle.load(f)
    f.close()
    
    return model

if __name__ == "__main__":

    # uncomment following codes to run it silently

    # STDIN = '/dev/null'
    # STDOUT = '/dev/null'
    # STDERR = '/dev/null'

    # try:
    #     pid = os.fork()
    #     if pid > 0:
    #         sys.exit(0)
    # except OSError:
    #     print('error #1')
    #     sys.exit(1)

    # os.umask(0)
    # os.setsid()

    # try:
    #     pid = os.fork()
    #     if pid > 0:
    #         sys.exit(0)
    # except OSError:
    #     print('error #2')
    #     sys.exit(1)

    # for f in sys.stdout, sys.stderr:
    #     f.flush()

    # si = open(STDIN, 'r')
    # so = open(STDOUT, 'a+')
    # se = open(STDERR, 'a+')
    # os.dup2(si.fileno(), sys.stdin.fileno())
    # os.dup2(so.fileno(), sys.stdout.fileno())
    # os.dup2(se.fileno(), sys.stderr.fileno())



    last_dic = {}
    cur_dic = {}
    features_dic = {}
    # Striping logs
    striping_dic ={}

    # Import predict model
    pred_model = import_model(model_file='DT_model.pkl')

    last_job_trace = open('./last_job_trace.result', 'a')
    if not last_job_trace:
        print('Cannot Open last_job_trace.result.')
        sys.exit()

    result_file = open('./total_striping_log.result', 'a')
    if not result_file:
        print('Cannot Open striping_log.result.')
        sys.exit()


    if is_null(file_name='total_striping_log.result') == False:

        with open('./total_striping_log.result', 'r') as f:
            total_striping_list = f.readlines()
        f.close()

        
        for line in total_striping_list:
            segs = line.strip().split()
            striping_dic[segs[0]] = [segs[1], segs[2], segs[3]]
        
    if is_null(file_name='./last_job_trace.result') == False:

        with open('./last_job_trace.result', 'r') as f:
            last_job_list = f.readlines()
        f.close()

        
        for line in last_job_list:
            segs = line.strip().split()
            last_dic[segs[0]] = [segs[1], segs[2], segs[3]]

        # Clear file
        last_job_trace.truncate(0)

    

    while True:
        queue = os.popen('squeue -h -o \"%i %u %U %C %V %j %D %Z\"').read()
        lines = queue.split("\n")

        for line in lines:
            segs = line.split()

            if len(segs) == 0:
                continue

            if segs[0] not in cur_dic and len(segs) == 8:

                abs_jobname = get_absolute_jobname(ori_jobname=segs[5])
                submit_timestamp = get_timestamp(segs[4])
                jobtype = cluster_jobname(ori_jobname=abs_jobname)
                pathtype = cluster_path(user_name=segs[1], path=segs[7])
                cur_dic[segs[0]] = [segs[2], segs[3], segs[4], segs[5], segs[6], segs[7]]
                features_dic[segs[0]] = [segs[2], segs[3], submit_timestamp, jobtype, segs[6], pathtype]


        for jobid in cur_dic:

            if jobid not in last_dic:
                upper_two_level_workdir = cur_dic[jobid][5].rsplit('/', 2)[0]
                upper_one_level_workdir = cur_dic[jobid][5].rsplit('/', 1)[0]
                current_workdir = cur_dic[jobid][5]

                result = pred_model.predict(np.array(features_dic[jobid]).reshape(1, -1))

                last_job_trace.write(jobid + ' ' + upper_two_level_workdir + ' ' + upper_one_level_workdir + ' ' + current_workdir + '\n')
                last_job_trace.flush()
                # Striping
                if result:
                    striping_dic[jobid] = [upper_two_level_workdir, upper_one_level_workdir, current_workdir]  

                    for dir_ in striping_dic[jobid]:
                        cmd = 'lfs setstripe ' + dir_ + ' -c 4'
                        os.system(cmd)
                        
                    # Write wordir list info to striping log
                    result_file.write(jobid + ' ' + upper_two_level_workdir + ' ' + upper_one_level_workdir + ' ' + current_workdir + '\n')
                    result_file.flush()
                    
        
        # Restore the initial state of the completed job
        pop_list = []
        for jobid in striping_dic:

            if jobid not in cur_dic:

                pop_list.append(jobid)
            
                # Striping workdir list   
                for dir_ in striping_dic[jobid]:
                    cmd = 'lfs setstripe ' + dir_ + ' -c 1'
                    os.system(cmd)
                    
        # Clear the completed job from striping logs
        for pop_id in pop_list:
            try:
                striping_dic.pop(pop_id)
            except:
                print(f'The Job ID {jobid} does not exist.')
        
        # Clear last job trace
        last_job_trace.truncate(0)

        last_dic = cur_dic
        cur_dic = {}
        features_dic = {}
        time.sleep(5)

    result_file.close()
    last_job_trace.close()
