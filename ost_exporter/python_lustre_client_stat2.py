from prometheus_client import start_http_server, Gauge
import random
import time
import os
import sys
import subprocess


lustre_root_dir = "/proc/fs/lustre/"

try:
    if sys.argv[1] == "ost":
        lustre_dir = "/proc/fs/lustre/obdfilter/"
    elif sys.argv[1] == "mds":
        lustre_dir = "/proc/fs/lustre/mdt/"
    else:
        print("python ./python_lustre_client_stat.py ost/mds [mds not available yet]")
        exit(0)
except:
    print("python ./python_lustre_client_stat.py ost/mds [mds not available yet]")
    exit(0)
else:
    pass


LUSTRE_CLIENT_STAT_EXPORTER = Gauge('lustre_client_stat2',
                                    'client statistics of lustre OSS or MDS',
                                    ['clientname', 'device', 'metrics_name'])

HOSTNAME_DIC = {}
def init_hostname_dic():
    with open("/etc/hosts") as hostfile:
        hostlines = hostfile.readlines()
    for line in hostlines:
        if line.startswith("#"):
            continue
        line_segs = line.split()
        if len(line_segs) < 2:
            continue
        host_ip = line_segs[0]
        host_name = line_segs[1]
        if not host_ip in HOSTNAME_DIC:
            HOSTNAME_DIC[host_ip] = host_name

def get_lustre_value():
    oss_list = os.listdir(lustre_dir)
    for oss_dir_index in range(0, len(oss_list)):
        oss_path = os.path.join(lustre_dir, oss_list[oss_dir_index])
        if os.path.isdir(oss_path):            
            getlustreclientstat(oss_list[oss_dir_index], oss_path)

def exporter_labels(devicename, hostname, type_, value):
    LUSTRE_CLIENT_STAT_EXPORTER.labels(
        device=devicename,
        clientname=hostname,
        metrics_name=type_).set(value)

def exporter_remove(label_values):
    """
    :param label_values: (devicename, hostname, type_) 
    """
    LUSTRE_CLIENT_STAT_EXPORTER.remove(*label_values)


def data_processing(devicename, hostname, temp_dic, all_dic):
    """
    :param devicename: ost_name
    :param hostname: node_name
    :param temp_dic: {node:[samples, bytes, status]}  status indicates whether data is available in the api
    :param all_dic: {ost:{node:[samples, bytes, status],...},...} temporary storage of full data
    :return: 0 or 1, 0 means the data is new, 1 means the {ost:{node:[...]}} exists
    """
    if devicename not in all_dic:                    
        all_dic[devicename] = temp_dic
        return 0        
    else:
        if hostname not in all_dic[devicename]:
            all_dic[devicename].update(temp_dic)
            return 0
        else:
            return 1


def api_data_update(devicename, hostname, statcontent, type_, all_dic):
    """
    :param devicename: ost_name
    :param hostname: node_name
    :param statcontent: raw data fields
    :param type_: write or read
    :param all_dic: {ost:{node:[samples, bytes, status],...},...} Temporary storage of full data
    :return:
    """
    temp_dic = {}
    temp_dic[hostname] = [statcontent[1], statcontent[6], 0]
    sample_ = 0
    bytes_ = 0
    flag = data_processing(devicename, hostname, temp_dic, all_dic)

    # 1 means that the data exists, but has not changed, need to be deleted
    if flag == 1:
        if all_dic[devicename][hostname][:2] == temp_dic[hostname][:2]:
            while all_dic[devicename][hostname][2] == 1:
                exporter_remove((hostname, devicename, type_ + "_bytes" ))
                exporter_remove((hostname, devicename, type_ + "_samples"))
                all_dic[devicename][hostname][2] = 0
        else:
            sample_ = eval(statcontent[1])
            bytes_ = eval(statcontent[6])
            all_dic[devicename][hostname][:2] = temp_dic[hostname][:2]
            
            if sample_ == 0 or bytes_ == 0:
                pass
            else:
                exporter_labels(devicename, hostname, type_ + "_bytes", bytes_)
                exporter_labels(devicename, hostname, type_ + "_samples", sample_)
                all_dic[devicename][hostname][2] = 1  


def getlustreclientstat(devicename, lustre_path):
    exportpath = os.path.join(lustre_path, "exports")
    exportslist = os.listdir(exportpath)
    lenexport = len(exportslist)
    for exportindex in range(0, lenexport):
        clientpath = os.path.join(exportpath, exportslist[exportindex])
        if "@" in clientpath:
            client_ip = exportslist[exportindex].split("@")[0]
            if not client_ip in HOSTNAME_DIC:
                continue
            hostname = HOSTNAME_DIC[client_ip]

            statpath = os.path.join(clientpath, "stats")
            with open(statpath) as statrecord:
                statlist = statrecord.readlines()
            for statline in statlist:
                statcontent = statline.split()
                if "write_bytes" in statcontent:
                    api_data_update(devicename, hostname, statcontent, "write", continuity_write)

                elif "read_bytes" in statcontent:
                    api_data_update(devicename, hostname, statcontent, "read", continuity_read)
                    
                else:
                    pass
        


if __name__ == '__main__':
    time.sleep(10)
    continuity_write = {}
    continuity_read = {}
    # Init /etc/hosts
    init_hostname_dic()
    # Start up the server to expose the metrics.
    start_http_server(9280)
    while True:
        try:
            get_lustre_value()
        except Exception as e:
            f = open("/var/log/lustre_exporter_python_ost_client_err.log", "a")
            f.write(str(e))
            f.close()
        time.sleep(1)


