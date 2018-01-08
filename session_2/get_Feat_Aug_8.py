#coding:utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def mape_object(y,d):

    g=1.0*np.sign(y-d)/d
    h=1.0/d
    return -g,h
def add_constact(df):
    return np.sum(1.0/df) / np.sum(1.0/df/df)
# 评价函数
def mape(y,d):
    c=d.get_label()
    result= np.sum(np.abs(y-c)/c)/len(c)
    return "mape",result

# 评价函数ln形式
def mape_ln(y,d):
    c=d.get_label()
    result= np.sum(np.abs(np.expm1(y)-np.abs(np.expm1(c)))/np.abs(np.expm1(c)))/len(c)
    return "mape",result

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

# training_data = pd.read_table(u'./quaterfinal_gy_cmp_training_traveltime.txt',sep=';')
# training_data.columns = ['link_ID', 'date', 'time_interval', 'travel_time']
# print training_data.head()
# print training_data.shape
# training_data = pd.merge(training_data,link_info,on='link_ID')

# testing_data = pd.read_table(u'./quaterfinal_gy_cmp_testing_template_seg1(update).txt',sep=';',header=None)
# testing_data.columns = ['link_ID', 'date', 'time_interval', 'travel_time']
# testing_data = pd.merge(testing_data,link_info,on='link_ID')
# testing_data['travel_time'] = np.NaN
# print testing_data.head()
# print testing_data.shape
# feature_date = pd.concat([training_data,testing_data],axis=0)

# feature_date = feature_date.sort_values(['link_ID','time_interval'])
# print feature_date
# feature_date.to_csv('./pre_data/feature_data.csv',index=False)

# feature_data = pd.read_csv('./pre_data/feature_data.csv')
# feature_data = feature_data[feature_data['date']>'2016-10-01']
# print feature_data
# feature_data_date = AddBaseTimeFeature(feature_data)
# print feature_data_date
# feature_data_date.to_csv('./pre_data/feature_data.csv',index=False)


# # test
# feature_data = pd.read_csv('./pre_data/feature_data.csv')
# test = feature_data.loc[(feature_data.time_interval_month == 6)&(feature_data.time_interval_begin_hour==8),: ]
# test.to_csv('./pre_data/test.csv',index=False)

import gc

from scipy.stats import mode
# 中位数
def mode_function(df):
    df = df.astype(int)
    counts = mode(df)
    return counts[0][0]

print u'8'
feature_data = pd.read_csv('./pre_data/feature_data.csv')
feature_data['link_ID'] = feature_data['link_ID'].astype(str)
# link_info_count = pd.read_csv('./pre_data/link_info_count.csv')
# link_info_count['link_ID'] = link_info_count['link_ID'].astype(str)
# feature_data = pd.merge(feature_data,link_info_count,on='link_ID',how='left')
# link_class = pd.get_dummies(feature_data['link_class'],prefix='link_class')
# int_count_onehot = pd.get_dummies(feature_data['in_count_'],prefix='in_count')
# out_count_onehot = pd.get_dummies(feature_data['out_count_'],prefix='out_count')
week = pd.get_dummies(feature_data['time_interval_week'],prefix='week')
# time_interval_minutes = pd.get_dummies(feature_data['time_interval_minutes'],prefix='time_interval_minutes')
# day = pd.get_dummies(feature_data['time_interval_day'],prefix='day')
feature_data.drop(['time_interval_week','link_class'],inplace=True,axis=1)
# linkId = pd.get_dummies(feature_data['link_ID'],prefix='link_id')
feature_data = pd.concat([feature_data,week],axis=1)
print feature_data.head()

train48 = feature_data.loc[(feature_data.time_interval_month == 4)&(feature_data.time_interval_begin_hour==8),: ]
for i in [58,48,38,28,18,8,0]:
    tmp = feature_data.loc[(feature_data.time_interval_month == 4)&(feature_data.time_interval_begin_hour == 7)&(feature_data.time_interval_minutes >= i),:]
    tmp = tmp.groupby(['link_ID', 'time_interval_day'])[
            'travel_time'].agg([('mean_%d' % (i), np.mean), ('median_%d' % (i), np.median),
                                ('mode_%d' % (i), mode_function), ('std_%d' % (i), np.std), ('max_%d' % (i), np.max),('min_%d' % (i), np.min)]).reset_index()
    tmp['std_%d' % (i)] = tmp['std_%d' % (i)].fillna(0)
    train48 = pd.merge(train48,tmp,on=['link_ID','time_interval_day'],how='left')

train58 = feature_data.loc[(feature_data.time_interval_month == 5)&(feature_data.time_interval_begin_hour==8),: ]
for i in [58,48,38,28,18,8,0]:
    tmp = feature_data.loc[(feature_data.time_interval_month == 5)&(feature_data.time_interval_begin_hour == 7)&(feature_data.time_interval_minutes >= i),:]
    tmp = tmp.groupby(['link_ID', 'time_interval_day'])[
            'travel_time'].agg([('mean_%d' % (i), np.mean), ('median_%d' % (i), np.median),
                                ('mode_%d' % (i), mode_function), ('std_%d' % (i), np.std), ('max_%d' % (i), np.max),('min_%d' % (i), np.min)]).reset_index()
    tmp['std_%d' % (i)] = tmp['std_%d' % (i)].fillna(0)
    train58 = pd.merge(train58,tmp,on=['link_ID','time_interval_day'],how='left')

train57 = feature_data.loc[(feature_data.time_interval_month == 5)&(feature_data.time_interval_begin_hour==7),: ]
for i in [58,48,38,28,18,8,0]:
    tmp = feature_data.loc[(feature_data.time_interval_month == 5)&(feature_data.time_interval_begin_hour == 6)&(feature_data.time_interval_minutes >= i),:]
    tmp = tmp.groupby(['link_ID', 'time_interval_day'])[
            'travel_time'].agg([('mean_%d' % (i), np.mean), ('median_%d' % (i), np.median),
                                ('mode_%d' % (i), mode_function), ('std_%d' % (i), np.std), ('max_%d' % (i), np.max),('min_%d' % (i), np.min)]).reset_index()
    tmp['std_%d' % (i)] = tmp['std_%d' % (i)].fillna(0)
    train57 = pd.merge(train57,tmp,on=['link_ID','time_interval_day'],how='left')

train518 = feature_data.loc[(feature_data.time_interval_month == 5)&(feature_data.time_interval_begin_hour==18),: ]
for i in [58,48,38,28,18,8,0]:
    tmp = feature_data.loc[(feature_data.time_interval_month == 5)&(feature_data.time_interval_begin_hour == 17)&(feature_data.time_interval_minutes >= i),:]
    tmp = tmp.groupby(['link_ID', 'time_interval_day'])[
            'travel_time'].agg([('mean_%d' % (i), np.mean), ('median_%d' % (i), np.median),
                                ('mode_%d' % (i), mode_function), ('std_%d' % (i), np.std), ('max_%d' % (i), np.max),('min_%d' % (i), np.min)]).reset_index()
    tmp['std_%d' % (i)] = tmp['std_%d' % (i)].fillna(0)
    train518 = pd.merge(train518,tmp,on=['link_ID','time_interval_day'],how='left')

train67 = feature_data.loc[(feature_data.time_interval_month == 6)&(feature_data.time_interval_begin_hour==7),: ]
for i in [58,48,38,28,18,8,0]:
    tmp = feature_data.loc[(feature_data.time_interval_month == 6)&(feature_data.time_interval_begin_hour == 6)&(feature_data.time_interval_minutes >= i),:]
    tmp = tmp.groupby(['link_ID', 'time_interval_day'])[
            'travel_time'].agg([('mean_%d' % (i), np.mean), ('median_%d' % (i), np.median),
                                ('mode_%d' % (i), mode_function), ('std_%d' % (i), np.std), ('max_%d' % (i), np.max),('min_%d' % (i), np.min)]).reset_index()
    tmp['std_%d' % (i)] = tmp['std_%d' % (i)].fillna(0)
    train67 = pd.merge(train67,tmp,on=['link_ID','time_interval_day'],how='left')

train418 = feature_data.loc[(feature_data.time_interval_month == 4)&(feature_data.time_interval_begin_hour==18),: ]
for i in [58,48,38,28,18,8,0]:
    tmp = feature_data.loc[(feature_data.time_interval_month == 4)&(feature_data.time_interval_begin_hour == 17)&(feature_data.time_interval_minutes >= i),:]
    tmp = tmp.groupby(['link_ID', 'time_interval_day'])[
            'travel_time'].agg([('mean_%d' % (i), np.mean), ('median_%d' % (i), np.median),
                                ('mode_%d' % (i), mode_function), ('std_%d' % (i), np.std), ('max_%d' % (i), np.max),('min_%d' % (i), np.min)]).reset_index()
    tmp['std_%d' % (i)] = tmp['std_%d' % (i)].fillna(0)
    train418 = pd.merge(train418,tmp,on=['link_ID','time_interval_day'],how='left')

# train47 = feature_data.loc[(feature_data.time_interval_month == 4)&(feature_data.time_interval_begin_hour==18),: ]
# for i in [58,48,38,28,18,8,0]:
#     tmp = feature_data.loc[(feature_data.time_interval_month == 4)&(feature_data.time_interval_begin_hour == 17)&(feature_data.time_interval_minutes >= i),:]
#     tmp = tmp.groupby(['link_ID', 'time_interval_day'])[
#             'travel_time'].agg([('mean_%d' % (i), np.mean), ('median_%d' % (i), np.median),
#                                 ('mode_%d' % (i), mode_function), ('std_%d' % (i), np.std), ('max_%d' % (i), np.max),('min_%d' % (i), np.min)]).reset_index()
#     tmp['std_%d' % (i)] = tmp['std_%d' % (i)].fillna(0)
#     train47 = pd.merge(train47,tmp,on=['link_ID','time_interval_day'],how='left')

train = pd.concat([train418,train67,train518,train57,train58,train48],axis=0)

train_history = feature_data.loc[(feature_data.time_interval_month == 4),: ]
train_history = train_history.groupby(['link_ID', 'time_interval_minutes'])[
            'travel_time'].agg([('mean_m', np.mean), ('median_m', np.median),
                                ('mode_m', mode_function), ('std_m', np.std), ('max_m', np.max),('min_m', np.min)]).reset_index()
# train_history['median_mode'] = 0.5 * train_history['mode_m'] + 0.5 * train_history['median_m']

train = pd.merge(train,train_history,on=['link_ID','time_interval_minutes'],how='left')
# train['speed_max'] = train['length']  / train['min_m']
# train['speed_min'] = train['length']  / train['max_m']
train['speed_mode'] = train['length']  / train['mode_m']
train['speed_median'] = train['length']  / train['median_m']

# train['120_speed'] = train['length']  / 120.0
train['mean_std'] = train['mean_m']  / train['std_m']
train['max_min_distance'] = train['max_m'] - train['min_m']

train_8 = feature_data.loc[(feature_data.time_interval_month == 4)&(feature_data.time_interval_begin_hour == 8),: ]
train_8 = train_8.groupby(['link_ID', 'time_interval_minutes'])[
            'travel_time'].agg([('median_8_', np.median)]).reset_index()

train = pd.merge(train,train_8,on=['link_ID','time_interval_minutes'],how='left')

print train.shape
train = train.fillna(-1)
train_label = np.log1p(train.pop('travel_time'))
# validation = feature_data.loc[(feature_data.time_interval_month == 5)&(feature_data.time_interval_begin_hour==8),: ]


test = feature_data.loc[(feature_data.time_interval_month == 6)&(feature_data.time_interval_begin_hour==8),: ]
for i in [58,48,38,28,18,8,0]:
    tmp = feature_data.loc[(feature_data.time_interval_month == 6)&(feature_data.time_interval_begin_hour == 7)&(feature_data.time_interval_minutes >= i),:]
    tmp = tmp.groupby(['link_ID', 'time_interval_day'])[
            'travel_time'].agg([('mean_%d' % (i), np.mean), ('median_%d' % (i), np.median),
                                ('mode_%d' % (i), mode_function), ('std_%d' % (i), np.std), ('max_%d' % (i), np.max),('min_%d' % (i), np.min)]).reset_index()
    tmp['std_%d' % (i)] = tmp['std_%d' % (i)].fillna(0)
    test = pd.merge(test,tmp,on=['link_ID','time_interval_day'],how='left')


test_history = feature_data.loc[(feature_data.time_interval_month == 5),: ]
test_history = test_history.groupby(['link_ID', 'time_interval_minutes'])[
            'travel_time'].agg([('mean_m', np.mean), ('median_m', np.median),
                                ('mode_m', mode_function), ('std_m', np.std), ('max_m', np.max),('min_m', np.min)]).reset_index()
# test_history['median_mode'] = 0.5 * test_history['mode_m'] + 0.5 * test_history['median_m']
test = pd.merge(test,test_history,on=['link_ID','time_interval_minutes'],how='left')

# test['speed_max'] = test['length']  / test['min_m']
# test['speed_min'] = test['length']  / test['max_m']
test['speed_mode'] = test['length']  / test['mode_m']
test['speed_median'] = test['length']  / test['median_m']

# test['120_speed'] = test['length']  / 120.0
test['mean_std'] = test['mean_m']  / test['std_m']
test['max_min_distance'] = test['max_m'] - test['min_m']

test_8 = feature_data.loc[(feature_data.time_interval_month == 5)&(feature_data.time_interval_begin_hour == 8),: ]
test_8 = test_8.groupby(['link_ID', 'time_interval_minutes'])[
            'travel_time'].agg([('median_8_', np.median)]).reset_index()

test = pd.merge(test,test_8,on=['link_ID','time_interval_minutes'],how='left')

print test.head()

# 缺失值的处理

test = test.fillna(-1)
test_label = np.log1p(test.pop('travel_time'))

train.drop(['link_ID','time_interval_begin_hour','time_interval_month','time_interval_begin','std_m','std_58','max_m'],inplace=True,axis=1)
test.drop(['link_ID','time_interval_begin_hour','time_interval_month','time_interval_begin','std_m','std_58','max_m'],inplace=True,axis=1)


import xgboost as xgb
# print xgb.__version__

xlf = xgb.XGBRegressor(max_depth=11,
                       learning_rate=0.005,
                       n_estimators=3000,
                       silent=True,
                       objective=mape_object,
                       gamma=0,
                       min_child_weight=5,
                       max_delta_step=0,
                       subsample=0.8,
                       colsample_bytree=0.8,
                       colsample_bylevel=1,
                       reg_alpha=1e0,
                       reg_lambda=0,
                       scale_pos_weight=1,
                       seed=9,
                       missing=None)

xlf.fit(train.values, train_label.values, eval_metric=mape_ln, verbose=True, eval_set=[(test.values, test_label.values)],early_stopping_rounds=3)
# xlf.fit(train.values, train_label.values, eval_metric=mape_ln, verbose=True, eval_set=[(test.values, test_label.values)],early_stopping_rounds=2)
print xlf.get_params()


sub = feature_data.loc[(feature_data.time_interval_month == 7)&(feature_data.time_interval_begin_hour==8),: ]
for i in [58,48,38,28,18,8,0]:
    tmp = feature_data.loc[(feature_data.time_interval_month == 7)&(feature_data.time_interval_begin_hour == 7)&(feature_data.time_interval_minutes >= i),:]
    tmp = tmp.groupby(['link_ID', 'time_interval_day'])[
            'travel_time'].agg([('mean_%d' % (i), np.mean), ('median_%d' % (i), np.median),
                                ('mode_%d' % (i), mode_function), ('std_%d' % (i), np.std), ('max_%d' % (i), np.max),('min_%d' % (i), np.min)]).reset_index()
    tmp['std_%d' % (i)] = tmp['std_%d' % (i)].fillna(0)
    sub = pd.merge(sub,tmp,on=['link_ID','time_interval_day'],how='left')

sub_history = feature_data.loc[(feature_data.time_interval_month == 5),: ]
sub_history = sub_history.groupby(['link_ID', 'time_interval_minutes'])[
            'travel_time'].agg([('mean_m', np.mean), ('median_m', np.median),
                                ('mode_m', mode_function), ('std_m', np.std), ('max_m', np.max),('min_m', np.min)]).reset_index()
# sub_history['median_mode'] = 0.5 * sub_history['mode_m'] + 0.5 * sub_history['median_m']

sub = pd.merge(sub,sub_history,on=['link_ID','time_interval_minutes'],how='left')
# sub['speed_max'] = sub['length'] / sub['min_m']
# sub['speed_min'] = sub['length'] / sub['max_m']
sub['speed_mode'] = sub['length'] / sub['mode_m']
sub['speed_median'] = sub['length'] / sub['median_m']

# sub['120_speed'] = sub['length'] / 120.0

sub['mean_std'] = sub['mean_m']  / sub['std_m']
sub['max_min_distance'] = sub['max_m'] - sub['min_m']

sub_history_8 = feature_data.loc[(feature_data.time_interval_month == 6)&(feature_data.time_interval_begin_hour == 8),: ]
sub_history_8 = sub_history_8.groupby(['link_ID', 'time_interval_minutes'])[
            'travel_time'].agg([('median_8_', np.median)]).reset_index()

sub = pd.merge(sub,sub_history_8,on=['link_ID','time_interval_minutes'],how='left')

print sub.head()

sub_label = np.log1p(sub.pop('travel_time'))

sub.drop(['link_ID','time_interval_begin_hour','time_interval_month','time_interval_begin','std_m','std_58','max_m'],inplace=True,axis=1)

result = xlf.predict(sub.values)

travel_time = pd.DataFrame({'travel_time':list(result)})
sub_demo = pd.read_table(u'./semifinal_gy_cmp_testing_template_seg2.txt',header=None,sep=';')

sub_demo.columns = ['link_ID','date','time_interval','travel_time']
sub_demo = sub_demo.sort_values(['link_ID','time_interval']).reset_index()
del sub_demo['index']
del sub_demo['travel_time']
tt = pd.concat([sub_demo,travel_time],axis=1)
# tt = tt.fillna(0)
tt['travel_time'] = np.round(np.expm1(tt['travel_time']),6)
tt[['link_ID','date','time_interval','travel_time']].to_csv('./2017-09-15_08_xgb.txt',sep='#',index=False,header=False)
print tt[['link_ID','date','time_interval','travel_time']].shape
print tt[['link_ID','date','time_interval','travel_time']].isnull().sum()


# [408]   validation_0-rmse:0.47515       validation_0-mape:0.298994
# [572]   validation_0-rmse:0.474525      validation_0-mape:0.298064
# [797]   validation_0-rmse:0.475478      validation_0-mape:0.297869
# 4 5 6 7
# [797]   validation_0-rmse:0.475478      validation_0-mape:0.297869