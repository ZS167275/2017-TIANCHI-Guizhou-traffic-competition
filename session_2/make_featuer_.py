#coding:utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def AddBaseTimeFeature(df):

    df['time_interval_begin'] = pd.to_datetime(df['time_interval'].map(lambda x: x[1:20]))
    df = df.drop(['date', 'time_interval'], axis=1)
    df['time_interval_month'] = df['time_interval_begin'].map(lambda x: x.strftime('%m'))
    # df['time_interval_year'] = df['time_interval_begin'].map(lambda x: x.strftime('%Y'))
    df['time_interval_day'] = df['time_interval_begin'].map(lambda x: x.day)
    df['time_interval_begin_hour'] = df['time_interval_begin'].map(lambda x: x.strftime('%H'))
    df['time_interval_minutes'] = df['time_interval_begin'].map(lambda x: x.strftime('%M'))
    # Monday=1, Sunday=7
    df['time_interval_week'] = df['time_interval_begin'].map(lambda x: x.weekday() + 1)
    return df


# txt => csv
# link_info = pd.read_table('./gy_contest_link_info.txt',sep=';')
# link_info = link_info.sort_values('link_ID')
# print link_info.dtypes
# link_info['link_ID'] = link_info['link_ID'].apply(str)

# session_1_training_data = pd.read_table(u'./session1/[新-训练集]gy_contest_traveltime_training_data_second.txt',sep=';')
# session_1_training_data.columns = ['link_ID', 'date', 'time_interval', 'travel_time']
# print session_1_training_data.dtypes
# session_1_training_data['link_ID'] = session_1_training_data['link_ID'].apply(str)

# session_1_training_data = session_1_training_data[(session_1_training_data['date']>='2017-03-01')&(session_1_training_data['date']<'2017-04-01')]
# print session_1_training_data.shape
# print session_1_training_data
# training_data = pd.read_table(u'./quaterfinal_gy_cmp_training_traveltime.txt',sep=';')
# training_data.columns = ['link_ID', 'date', 'time_interval', 'travel_time']
# print training_data.dtypes
# training_data['link_ID'] = training_data['link_ID'].apply(str)
# print training_data.head()
# print training_data.shape

# training_data = pd.concat([session_1_training_data,training_data],axis=0)
# print training_data.head()
# print training_data.shape

# training_data = pd.merge(training_data,link_info,on='link_ID')

# testing_data = pd.read_table(u'./semifinal_gy_cmp_testing_template_seg2.txt',sep=';',header=None)
# testing_data.columns = ['link_ID', 'date', 'time_interval', 'travel_time']
# # testing_data.columns = ['link_ID', 'date', 'time_interval', 'travel_time']
# print testing_data.dtypes
# testing_data['link_ID'] = testing_data['link_ID'].apply(str)
# testing_data = pd.merge(testing_data,link_info,on='link_ID')
# testing_data['travel_time'] = np.NaN
# print testing_data.head()
# print testing_data.shape
# print testing_data.isnull().sum()
# feature_date = pd.concat([training_data,testing_data],axis=0)

# feature_date = feature_date.sort_values(['link_ID','time_interval'])
# print feature_date
# print feature_date.dtypes
# feature_date['link_ID'] = feature_date['link_ID'].apply(str)
# feature_date.to_csv('./pre_data/feature_data.csv',index=False)
# print feature_date.dtypes

feature_data = pd.read_csv('./pre_data/feature_data.csv',dtype='str')
print feature_data
feature_data = feature_data[feature_data['date']>'2016-10-01']
print feature_data.dtypes
feature_data['link_ID'] = feature_data['link_ID'].apply(str)
print feature_data
feature_data_date = AddBaseTimeFeature(feature_data)
print feature_data_date
feature_data_date.to_csv('./pre_data/feature_data.csv',index=False)


# test
# feature_data = pd.read_csv('./pre_data/feature_data.csv')
# test = feature_data.loc[(feature_data.time_interval_month == 6)&(feature_data.time_interval_begin_hour==8),: ]
# test.to_csv('./pre_data/test.csv',index=False)