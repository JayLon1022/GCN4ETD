#coding=utf-8
import scipy.sparse as sp
import os
import pandas as pd
from FlowFeature import global_path
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from collections import Counter
import time
from sklearn.model_selection import cross_val_score  # 交叉验证所需的函数
from sklearn.ensemble import RandomForestClassifier
np.random.seed(222)

def tcp_tuple(path):

    files = os.listdir(path)
    # os.popen('D:')
    # os.popen('cd D:\Program Files\Wireshark')
    cmd = 'tshark -r {pcap} -q -z conv,tcp'
    features_columns = ['Duration', 'Source Port', 'Destination Port', 'Packets', 'Bytes']
    headers = [features_columns]
    sum_df = pd.DataFrame(headers)


    # 获取所有pcap文件内的数据

    for file in files:
        pcap = path + '\\' + file
        data = os.popen(cmd.format(pcap=pcap)).read()   #获取命令执行结果
        data = np.array(data.split('|')[-1].split('=')[0].split())  #切割出需要的数据内容
        df = pd.DataFrame(data.reshape(-1, 11)) #重新排列，转为csv格式
        df = df.drop([1, 7, 8, 9], axis=1)  #筛除无用数据
        df = df.T.reset_index(drop=True).T  #重置列索引
        sum_df = pd.concat([sum_df, df])    #拼接进总的DataFrame
        print(file, ' 提取完毕')

    return sum_df



def tuple_csv(DataFrame, path):

    label_scan44 = ['label']
    label_background = ['label']
    label_blacklist = ['label']
    label_dos = ['label']
    label_nerisbotnet = ['label']
    label_anomaly_spam = ['label']
    label_scan11 = ['label']

    if path == global_path.white_path:
        for _ in range(DataFrame.shape[0] - 1):
            label_background.append('0')
        DataFrame['7'] = label_white

    if path == global_path.black_path:
        for _ in range(DataFrame.shape[0] - 1):
            label_dos.append('1')
        DataFrame['7'] = label_dos

    elif path == global_path.black_path:
        for _ in range(DataFrame.shape[0] - 1):
            label_blacklist.append('2')
        DataFrame['7'] = label_blacklist

    elif path == global_path.black_path:
        for _ in range(DataFrame.shape[0] - 1):
            label_scan44.append('3')
        DataFrame['7'] = label_scan44

    elif path == global_path.black_path:
        for _ in range(DataFrame.shape[0] - 1):
            label_nerisbotnet.append('4')
        DataFrame['7'] = label_nerisbotnet

    elif path == global_path.black_path:
        for _ in range(DataFrame.shape[0] - 1):
            label_anomaly_spam.append('5')
        DataFrame['7'] = label_anomaly_spam

    elif path == global_path.black_path:
        for _ in range(DataFrame.shape[0] - 1):
            label_scan11.append('1')
        DataFrame['7'] = label_scan11

    DataFrame = DataFrame.drop([0], axis=0)
    DataFrame.columns = ['Duration', 'Source Port', 'Destination Port', 'Packets', 'Bytes']
    print(DataFrame)
    if os.path.exists(tuple_data_path):
        DataFrame.to_csv(tuple_data_path, index=False, header=False, mode='a')
    else:
        DataFrame.to_csv(tuple_data_path, index=False)


if __name__ == '__main__':
    tuple_data_path = 'processed_data.csv'
    if not os.path.exists(tuple_data_path):
        white_df = tcp_tuple(global_path.white_path)
        tuple_csv(white_df, global_path.white_path)

        black_df = tcp_tuple(global_path.black_path)
        tuple_csv(black_df, global_path.black_path)

    df = pd.read_csv(tuple_data_path)
    print(df['label'].value_counts())
