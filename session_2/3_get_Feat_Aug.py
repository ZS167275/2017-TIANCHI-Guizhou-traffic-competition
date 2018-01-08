#coding:utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def mape_object(y,d):

    g=1.0*np.sign(y-d)/d
    h=1.0/d
    return -g,h

# 评价函数
def mape(y,d):
    c=d.get_label()
    result=np.sum(np.abs(y-c)/c)/len(c)
    return "mape",result

# 评价函数ln形式
def mape_ln(y,d):
    c=d.get_label()
    result=np.sum(np.abs(np.expm1(y)-np.abs(np.expm1(c)))/np.abs(np.expm1(c)))/len(c)
    return "mape",result

def AddBaseTimeFeature(df):

    df['time_interval_begin'] = pd.to_datetime(df['time_interval'].map(lambda x: x[1:20]))
    df = df.drop(['date', 'time_interval'], axis=1)
    df['time_interval_month'] = df['time_interval_begin'].map(lambda x: x.strftime('%m'))
    df['time_interval_day'] = df['time_interval_begin'].map(lambda x: x.day)
    df['time_interval_begin_hour'] = df['time_interval_begin'].map(lambda x: x.strftime('%H'))
    df['time_interval_minutes'] = df['time_interval_begin'].map(lambda x: x.strftime('%M'))
    # Monday=1, Sunday=7
    df['time_interval_week'] = df['time_interval_begin'].map(lambda x: x.weekday() + 1)
    return df

# gy_contest_link_top.txt
# link_top = pd.read_table('./gy_contest_link_top.txt',sep=';')
# print link_top
# 4377906284594800514                                4377906284514600514

'''
link_info = pd.read_table('./gy_contest_link_info.txt',sep=';')
link_info = link_info.sort_values('link_ID')
training_data = pd.read_table('./gy_contest_link_traveltime_training_data.txt',sep=';')
print training_data.shape
training_data = pd.merge(training_data,link_info,on='link_ID')
testing_data = pd.read_table('./sub_demo.txt',sep='#',header=None)
testing_data.columns = ['link_ID', 'date', 'time_interval', 'travel_time']
testing_data = pd.merge(testing_data,link_info,on='link_ID')
testing_data['travel_time'] = np.NaN
print testing_data.shape
feature_date = pd.concat([training_data,testing_data],axis=0)
feature_date = feature_date.sort_values(['link_ID','time_interval'])
print feature_date
feature_date.to_csv('./pre_data/feature_data.csv',index=False)

feature_data = pd.read_csv('./pre_data/feature_data.csv')
feature_data_date = AddBaseTimeFeature(feature_data)
print feature_data_date
feature_data_date.to_csv('./pre_data/feature_data.csv',index=False)
'''
# txt => csv



'''
# test
feature_data = pd.read_csv('./pre_data/feature_data.csv')
test = feature_data.loc[(feature_data.time_interval_month == 6)&(feature_data.time_interval_begin_hour==8),: ]
test.to_csv('./pre_data/test.csv',index=False)
'''


from scipy.stats import mode
# 中位数
def mode_function(df):
    counts = mode(df)
    return counts[0][0]

feature_data = pd.read_csv('./pre_data/feature_data.csv')
feature_data['link_ID'] = feature_data['link_ID'].astype(str)
week = pd.get_dummies(feature_data['time_interval_week'],prefix='week')
del feature_data['time_interval_week']
feature_data = pd.concat([feature_data,week],axis=1)

train = feature_data.loc[(feature_data.time_interval_month == 5)&(feature_data.time_interval_begin_hour==8),: ]
for i in [58,48,38,28,18,8,0]:
    tmp = feature_data.loc[(feature_data.time_interval_month == 5)&(feature_data.time_interval_begin_hour == 7)&(feature_data.time_interval_minutes >= i),:]
    tmp = tmp.groupby(['link_ID', 'time_interval_day'])[
            'travel_time'].agg([('mean_%d' % (i), np.mean), ('median_%d' % (i), np.median),
                                ('mode_%d' % (i), mode_function), ('std_%d' % (i), np.std), ('max_%d' % (i), np.max),('min_%d' % (i), np.min)]).reset_index()
    train = pd.merge(train,tmp,on=['link_ID','time_interval_day'],how='left')


train_history = feature_data.loc[(feature_data.time_interval_month == 4),: ]
train_history = train_history.groupby(['link_ID', 'time_interval_minutes'])[
            'travel_time'].agg([('mean_m', np.mean), ('median_m', np.median),
                                ('mode_m', mode_function), ('std_m', np.std), ('max_m', np.max),('min_m', np.min)]).reset_index()

train = pd.merge(train,train_history,on=['link_ID','time_interval_minutes'],how='left')
# print train.head()

train_label = np.log1p(train.pop('travel_time'))
# validation = feature_data.loc[(feature_data.time_interval_month == 5)&(feature_data.time_interval_begin_hour==8),: ]


test = feature_data.loc[(feature_data.time_interval_month == 6)&(feature_data.time_interval_begin_hour==8),: ]
for i in [58,48,38,28,18,8,0]:
    tmp = feature_data.loc[(feature_data.time_interval_month == 6)&(feature_data.time_interval_begin_hour == 7)&(feature_data.time_interval_minutes >= i),:]
    tmp = tmp.groupby(['link_ID', 'time_interval_day'])[
            'travel_time'].agg([('mean_%d' % (i), np.mean), ('median_%d' % (i), np.median),
                                ('mode_%d' % (i), mode_function), ('std_%d' % (i), np.std), ('max_%d' % (i), np.max),('min_%d' % (i), np.min)]).reset_index()
    test = pd.merge(test,tmp,on=['link_ID','time_interval_day'],how='left')


test_history = feature_data.loc[(feature_data.time_interval_month == 5),: ]
test_history = test_history.groupby(['link_ID', 'time_interval_minutes'])[
            'travel_time'].agg([('mean_m', np.mean), ('median_m', np.median),
                                ('mode_m', mode_function), ('std_m', np.std), ('max_m', np.max),('min_m', np.min)]).reset_index()

test = pd.merge(test,test_history,on=['link_ID','time_interval_minutes'],how='left')
# print test.head()

test_label = np.log1p(test.pop('travel_time'))

train.drop(['time_interval_begin_hour','time_interval_month','time_interval_begin'],inplace=True,axis=1)
test.drop(['time_interval_begin_hour','time_interval_month','time_interval_begin'],inplace=True,axis=1)

# validation_0-mape:0.346155 0.60

import xgboost as xgb

xlf = xgb.XGBRegressor(max_depth=11,
                       learning_rate=0.01,
                       n_estimators=1000,
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


xlf.fit(train.values, train_label.values, eval_metric=mape_ln, verbose=True, eval_set=[(test.values, test_label.values)],early_stopping_rounds=2)
# print xlf.get_params()


sub = feature_data.loc[(feature_data.time_interval_month == 7)&(feature_data.time_interval_begin_hour==8),: ]
for i in [58,48,38,28,18,8,0]:
    tmp = feature_data.loc[(feature_data.time_interval_month == 7)&(feature_data.time_interval_begin_hour == 7)&(feature_data.time_interval_minutes >= i),:]
    tmp = tmp.groupby(['link_ID', 'time_interval_day'])[
            'travel_time'].agg([('mean_%d' % (i), np.mean), ('median_%d' % (i), np.median),
                                ('mode_%d' % (i), mode_function), ('std_%d' % (i), np.std), ('max_%d' % (i), np.max),('min_%d' % (i), np.min)]).reset_index()
    sub = pd.merge(sub,tmp,on=['link_ID','time_interval_day'],how='left')

sub_history = feature_data.loc[(feature_data.time_interval_month == 6),: ]
sub_history = sub_history.groupby(['link_ID', 'time_interval_minutes'])[
            'travel_time'].agg([('mean_m', np.mean), ('median_m', np.median),
                                ('mode_m', mode_function), ('std_m', np.std), ('max_m', np.max),('min_m', np.min)]).reset_index()

sub = pd.merge(sub,sub_history,on=['link_ID','time_interval_minutes'],how='left')
# print sub.head()

sub_label = np.log1p(sub.pop('travel_time'))

sub.drop(['time_interval_begin_hour','time_interval_month','time_interval_begin'],inplace=True,axis=1)

result = xlf.predict(sub.values)

travel_time = pd.DataFrame({'travel_time':list(result)})
sub_demo = pd.read_table(u'./quaterfinal_gy_cmp_testing_template_seg1(update).txt',header=None,sep=';')

sub_demo.columns = ['link_ID','date','time_interval','travel_time']
sub_demo = sub_demo.sort_values(['link_ID','time_interval']).reset_index()
del sub_demo['index']

del sub_demo['travel_time']

tt = pd.concat([sub_demo,travel_time],axis=1)
# tt = tt.fillna(0)
tt['travel_time'] = np.round(np.expm1(tt['travel_time']),6)
tt[['link_ID','date','time_interval','travel_time']].to_csv('./ss_2017-09-05.txt',sep='#',index=False,header=False)
print tt[['link_ID','date','time_interval','travel_time']].shape
print tt[['link_ID','date','time_interval','travel_time']].isnull().sum()

# mapodoufu1 = pd.read_table('./ss_2017-08-11.txt',header=None,sep='#')
# mapodoufu2 = pd.read_table('./siyueshinidehuangyan_2017-08-10.txt',header=None,sep='#')
#
#
# print sum(mapodoufu1[0]==mapodoufu2[0])
# print sum(mapodoufu1[1]==mapodoufu2[1])
# print sum(mapodoufu1[2]==mapodoufu2[2])
# print sum(mapodoufu1[3]==mapodoufu2[3])
# result=np.sum(np.abs(mapodoufu1[3]-mapodoufu2[3])/mapodoufu2[3])/len(mapodoufu2[3])
# print result

# [374]	validation_0-rmse:0.443417	validation_0-mape:0.295266 0.0795302055463