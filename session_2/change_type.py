#coding:utf-8

import pandas as pd
import numpy as np
# my_para08 = pd.read_csv('./analy_data_08.txt')
# my_para15 = pd.read_csv('./analy_data_15.txt')
# my_para18 = pd.read_csv('./analy_data_18.txt')
# xx = pd.read_csv('./pre_data/feature_data_1.csv',dtype='str')
#
# x1 = xx[(xx['time_interval_month']=='06')&(xx['time_interval_begin_hour']=='08')].reset_index()
# x2 = xx[(xx['time_interval_month']=='06')&(xx['time_interval_begin_hour']=='15')].reset_index()
# x3 = xx[(xx['time_interval_month']=='06')&(xx['time_interval_begin_hour']=='18')].reset_index()
#
# x1 = pd.concat([x1[['link_ID','time_interval_begin_hour']],my_para08],axis=1)
# x2 = pd.concat([x2[['link_ID','time_interval_begin_hour']],my_para15],axis=1)
# x3 = pd.concat([x3[['link_ID','time_interval_begin_hour']],my_para18],axis=1)
#
# print x1
# print x2
# print x3
# del x1['Unnamed: 0']
# del x2['Unnamed: 0']
# del x3['Unnamed: 0']
#
# x1.to_csv('./06.txt')
# x2.to_csv('./15.txt')
# x3.to_csv('./18.txt')

my_para08 = pd.read_table('./quaterfinal_gy_cmp_training_traveltime.txt',sep=';')
my_para08['link_ID'] = my_para08['link_ID'].apply(str)
print my_para08['link_ID'].unique