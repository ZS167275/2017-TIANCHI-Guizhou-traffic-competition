#coding:utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

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

# 基础时间特征
def AddBaseTimeFeature(df):
    df['time_interval_begin'] = pd.to_datetime(df['time_interval'].map(lambda x: x[1:20]))
    # train_data['time_interval_end'] = pd.to_datetime(train_data['time_interval'].map(lambda x : x[21:-1]))
    # 删除 data time_interval link_class
    df = df.drop(['date', 'time_interval', 'link_class'], axis=1)
    print df.columns
    # 小时 分钟 月 日 星期
    df['time_interval_month'] = df['time_interval_begin'].map(lambda x: x.strftime('%m'))
    df['time_interval_day'] = df['time_interval_begin'].map(lambda x: x.day)
    df['time_interval_begin_hour'] = df['time_interval_begin'].map(lambda x: x.strftime('%H'))
    df['time_interval_minutes'] = df['time_interval_begin'].map(lambda x: x.strftime('%M'))
    # Monday=1, Sunday=7
    df['time_interval_week'] = df['time_interval_begin'].map(lambda x: x.weekday() + 1)
    del df['time_interval_begin']
    return df

#  设置节假日的信息
def AddHolidayAndWeekOneHotFeature(df):
    df['holiday'] = 0
    df.loc[(df['time_interval_month']==4)&(df['time_interval_day'].isin(list([2,3,4]))),'holiday'] = 1
    df.loc[(df['time_interval_month']==5)&(df['time_interval_day'].isin(list([1,2]))),'holiday'] = 1
    df.loc[(df['time_interval_month']==6)&(df['time_interval_day'].isin(list([9,10,11]))),'holiday'] = 1
    # 由于12日属于正常上班因此当作周一处理
    df.loc[(df['time_interval_month']==6)&(df['time_interval_day'].isin(list([12]))),'time_interval_week'] = 1
    df_week = pd.get_dummies(df['time_interval_week'],prefix='week')
    df.drop(['time_interval_week'],inplace=True,axis=1)
    df = pd.concat([df,df_week],axis=1)
    return df

# 统计特征
# 路段在当前月份一天内，30个时段的均值，方差，中位数，众数
def Count_OneDay_Feat(df):
    count_feature = df.groupby(
        ['link_ID', 'time_interval_month', 'time_interval_day', 'time_interval_minutes'])[
        'travel_time'].agg([('mean_', np.mean), ('median_', np.median), ('mode_', mode_function),('std_',np.std),('max_',np.max),('min_',np.min)]).reset_index()

    df = pd.merge(df, count_feature,
                     on=['link_ID', 'time_interval_month', 'time_interval_day', 'time_interval_minutes'], how='left')
    return df

from scipy.stats import mode
# 中位数
def mode_function(df):
    counts = mode(df)
    return counts[0][0]
    # return np.argmax(counts)

def All_month_count_feat(df):
    df_count = df.groupby(['link_ID', 'time_interval_month', 'time_interval_minutes'])[
        'travel_time'].agg([('mean_m', np.mean), ('median_m', np.median), ('mode_m', mode_function), ('max_m', np.max),
                            ('min_m', np.min),('std_m',np.std)]).reset_index()
    df = pd.merge(df, df_count,
                              on=['link_ID', 'time_interval_minutes', 'time_interval_month'])
    return df
'''
# 已给出的数据中不存在nan值
# link_class 全为 1 
# travel_time max 1.965600e+03 2.000000e-01 绘制分布图 log1p分布
# 读取道路属性
link_info = pd.read_table('./gy_contest_link_info.txt',sep=';')
# 读取历史数据
training_data = pd.read_table('./gy_contest_link_traveltime_training_data.txt',sep=';')
# 历史数据 + 道路属性
train_data = pd.merge(training_data,link_info,on='link_ID',how='left')
print train_data.shape
# 历史数据 + 道路属性 .csv
train_data.to_csv('./data/train_data.csv',index=False)

'''

'''
# 基础时间处理

train_data = pd.read_csv('./data/train_data.csv')
train_data = AddBaseTimeFeature(train_data)
train_data.to_csv('./data/train_test.csv',index=False)

'''

'''
每个月份的记录个数
3    2502906 0-23点数据
4    2392776 0-23点数据
5    2567206 0-23点数据
6     199496 预测月份 给了 6-7 点数据
count 7662384

u'link_ID', u'travel_time', u'length', u'width', u'time_interval_month',u'time_interval_day', 
u'time_interval_begin_hour',u'time_interval_minutes', u'time_interval_week', u'log_travel_time'
计算法定节假日：
4月 2 3 4
5月 1 2
6月 9 10 11 6月12日上班

AddHolidayAndWeekOneHotFeature() 
index list = [u'link_ID', u'travel_time', u'length', u'width', u'time_interval_month',
       u'time_interval_day', u'time_interval_begin_hour',
       u'time_interval_minutes', u'holiday', u'week_1', u'week_2', u'week_3',
       u'week_4', u'week_5', u'week_6', u'week_7']
       
train = pd.read_csv('./data/train.csv')
train = AddHolidayAndWeekOneHotFeature(train)
train = Count_OneDay_Feat(train)
train.to_csv('./data/train_org_feat.csv',index=False)
print train.shape
       
'''

'''
Index([u'link_ID', u'travel_time', u'length', u'width', u'time_interval_month',
       u'time_interval_day', u'time_interval_begin_hour',
       u'time_interval_minutes', u'holiday', u'week_1', u'week_2', u'week_3',
       u'week_4', u'week_5', u'week_6', u'week_7', u'mean_', u'median_',
       u'mode_', u'std_'],
       

'''
#########################构造提交数据##################################
'''
# 读取道路属性
link_info = pd.read_table('./gy_contest_link_info.txt',sep=';')
# 读取历史数据
testing_data = pd.read_table('./mapodoufu_2017-07-27.txt',sep='#',header=None)
testing_data.columns = ['link_ID','date','time_interval','travel_time']
# 历史数据 + 道路属性
test_data = pd.merge(testing_data,link_info,on='link_ID',how='left')
print test_data.shape
# 历史数据 + 道路属性 .csv
test_data.to_csv('./data/test_data.csv',index=False)
'''

'''
# 基础时间处理

test_data = pd.read_csv('./data/test_data.csv')
test_data = AddBaseTimeFeature(test_data)
test_data.to_csv('./data/test.csv',index=False)
'''

'''
test = pd.read_csv('./data/test.csv')
test = AddHolidayAndWeekOneHotFeature(test)
test = Count_OneDay_Feat(test)
test.to_csv('./data/test_org_feat.csv',index=False)
print test.shape
'''



'''
# 构造整个月份的特征
train = pd.read_csv('./data/train.csv')
train = train.fillna(0)
train = All_month_count_feat(train)
print train
train.to_csv('./data/train_org_feat_change.csv',index=False)
'''

train_org_feat = pd.read_csv('./data/train_org_feat.csv')
train_org_feat = train_org_feat.fillna(0)

train_org_feat_change = pd.read_csv('./data/train_org_feat_change.csv')
train_org_feat_change = train_org_feat_change.fillna(0)


# # 选取5月份数据的8-9点时间段构造测试数据 5月份所有8点时段数据
test = train_org_feat[train_org_feat['time_interval_month']==5]
test = test[test['time_interval_begin_hour']==8]
print test.shape
test.drop(['mean_','median_','mode_','std_','max_','min_','time_interval_month','time_interval_begin_hour'],inplace=True,axis=1)
# 验证的标签数据
test_label = test.pop('travel_time')
test_label = np.log1p(test_label)
# 获取7-8点的统计特征数据
count_test = train_org_feat_change[train_org_feat_change['time_interval_month']==4]
count_test = count_test[(count_test['time_interval_begin_hour']==8)]
# last_test = count_test[count_test['time_interval_minutes']==58]['travel_time']
count_test = count_test[['link_ID','travel_time','time_interval_minutes','time_interval_day','mean_m','median_m','mode_m','std_m','max_m','min_m'
                         ]]
count_test['mode_median_m'] = (count_test['median_m'] + count_test['mode_m'])/2.0
count_test['travel_time'] = np.log1p(count_test['travel_time'])
test = pd.merge(test,count_test,on=['link_ID','time_interval_minutes','time_interval_day'],how='left')

count_test = train_org_feat[train_org_feat['time_interval_month']==4]
count_test = count_test[(count_test['time_interval_begin_hour']==8)]
count_test = count_test[['link_ID','time_interval_minutes','time_interval_day','mean_','median_','mode_','std_','max_','min_'
                         ]]
count_test['mode_median_'] = (count_test['mode_'] + count_test['median_'])/2.0
test = pd.merge(test,count_test,on=['link_ID','time_interval_minutes','time_interval_day'],how='left')
test = test.fillna(test.median())
# print test.head()
# print test.shape



train = train_org_feat[train_org_feat['time_interval_month']==4]
train = train[train['time_interval_begin_hour']==8]
print train.shape
train.drop(['mean_','median_','mode_','std_','max_','min_','time_interval_month','time_interval_begin_hour'],inplace=True,axis=1)
# 验证的标签数据
train_label = train.pop('travel_time')
train_label = np.log1p(train_label)
# 获取7-8点的统计特征数据
count_train = train_org_feat_change[train_org_feat_change['time_interval_month']==3]
count_train = count_train[(count_train['time_interval_begin_hour']==8)]
# last_train = count_train[count_train['time_interval_minutes']==58]['travel_time']
count_train = count_train[['link_ID','travel_time','time_interval_minutes','time_interval_day','mean_m','median_m','mode_m','std_m','max_m','min_m'
                           ]]
count_train['mode_median_m'] = (count_train['median_m'] + count_train['mode_m'])/2.0
count_train['travel_time'] = np.log1p(count_train['travel_time'])
train = pd.merge(train,count_train,on=['link_ID','time_interval_minutes','time_interval_day'],how='left')


count_train = train_org_feat[train_org_feat['time_interval_month']==3]
count_train = count_train[(count_train['time_interval_begin_hour']==8)]
count_train = count_train[['link_ID','time_interval_minutes','time_interval_day','mean_','median_','mode_','std_','max_','min_'
                           ]]
count_train['mode_median_'] = (count_train['mean_'] + count_train['mode_'])/2.0
train = pd.merge(train,count_train,on=['link_ID','time_interval_minutes','time_interval_day'],how='left')
train = train.fillna(train.median())
# print train.head()
# print train.shape

feat = ['length','width','time_interval_minutes','travel_time',
        'mean_m', 'median_m','mode_m','max_m','min_m','std_m',
        'mean_','median_','mode_','std_','week_1','week_5','week_6','week_7']

# [29]	validation_0-rmse:0.635224	validation_0-mape:0.376652
# [29]	validation_0-rmse:0.680168	validation_0-mape:0.378365
# [29]	validation_0-rmse:0.691687	validation_0-mape:0.375677
# [309]	validation_0-rmse:0.683436	validation_0-mape:0.36668 9000 0.239716016874
# [304]	validation_0-rmse:0.68838	validation_0-mape:0.367605 12 0.241281228054
# [294]	validation_0-rmse:0.700439	validation_0-mape:0.369061 0.2386256583
# model 模型
from sklearn.ensemble import GradientBoostingRegressor

alpha = 0.95

clf = GradientBoostingRegressor(loss='quantile', alpha=alpha,
                                n_estimators=1000, max_depth=11,
                                learning_rate=.1, min_samples_leaf=9,
                                min_samples_split=9)


# clf.set_params(alpha=1.0 - alpha)
clf.set_params(loss='ls')
clf.fit(train[feat].values, train_label.values)
pre = clf.predict(test[feat].values)
result=np.sum(np.abs(np.expm1(pre)-np.abs(np.expm1(test_label.values)))/np.abs(np.expm1(test_label.values)))/len(test_label.values)
print result


# xlf = xgb.XGBRegressor(max_depth=11,
#                        learning_rate=0.01,
#                        n_estimators=1000,
#                        silent=True,
#                        objective=mape_object,
#                        gamma=0,
#                        min_child_weight=5,
#                        max_delta_step=0,
#                        subsample=0.8,
#                        colsample_bytree=0.8,
#                        colsample_bylevel=1,
#                        reg_alpha=1e0,
#                        reg_lambda=0,
#                        scale_pos_weight=1,
#                        seed=666,
#                        missing=None)
#
# xlf.fit(train[feat].values, train_label.values, eval_metric=mape_ln, verbose=True, eval_set=[(test[feat].values, test_label.values)],early_stopping_rounds=2)
# limit = xlf.best_iteration + 1
# print xlf.get_params()
#
# ##################################################################
# sub_org_feat = pd.read_csv('./data/test_org_feat.csv')
# sub_org_feat = sub_org_feat.fillna(0)
#
# # sub = sub_org_feat
# sub = sub_org_feat[sub_org_feat['time_interval_month']==6]
# sub = sub[sub['time_interval_begin_hour']==8]
# print sub.shape
# sub.drop(['mean_','median_','mode_','std_','max_','min_','time_interval_month','time_interval_begin_hour'],inplace=True,axis=1)
#
# # 验证的标签数据
# sub_label = sub.pop('travel_time')
# sub_label = np.log1p(sub_label)
# # 获取7-8点的统计特征数据
# count_sub = train_org_feat_change[train_org_feat_change['time_interval_month']==5]
# count_sub = count_sub[(count_sub['time_interval_begin_hour']==8)]
#
# count_sub = count_sub[['link_ID','travel_time','time_interval_minutes','time_interval_day','mean_m','median_m','mode_m','std_m','max_m','min_m'
#                            ]]
# count_sub['mode_median_m'] = (count_sub['mean_m'] + count_sub['mode_m'])/2.0
# count_sub['travel_time'] = np.log1p(count_sub['travel_time'])
#
# sub = pd.merge(sub,count_sub,on=['link_ID','time_interval_minutes','time_interval_day'],how='left')
#
#
# count_sub = train_org_feat[train_org_feat['time_interval_month']==5]
# count_sub = count_sub[(count_sub['time_interval_begin_hour']==8)]
# count_sub = count_sub[['link_ID','time_interval_minutes','time_interval_day','mean_','median_','mode_','std_','max_','min_'
#                            ]]
# count_sub['mode_median_'] = (count_sub['mean_'] + count_sub['mode_'])/2.0
# sub = pd.merge(sub,count_sub,on=['link_ID','time_interval_minutes','time_interval_day'],how='left')
# sub = sub.fillna(train.median())
# print sub.head()
# print sub
# print sub.shape
#
# result = xlf.predict(sub[feat].values)
# print len(list(result))
# sub_demo = pd.read_table('./sub_demo.txt',header=None,sep='#')
# sub_demo.columns = ['link_ID','date','time_interval','travel_time']
# del sub_demo['travel_time']
# travel_time = pd.DataFrame({'travel_time':list(result)})
# tt = pd.concat([sub_demo,travel_time],axis=1)
# # tt = tt.fillna(0)
# tt['travel_time'] = np.round(np.expm1(tt['travel_time']),6)
# tt[['link_ID','date','time_interval','travel_time']].to_csv('./mapodoufu_2017-08-01.txt',sep='#',index=False,header=False)
# print tt[['link_ID','date','time_interval','travel_time']].shape
# print tt[['link_ID','date','time_interval','travel_time']].isnull().sum()
#
#
# # 该路段上游路段数量，该路段下游路段数量
#
# mapodoufu1 = pd.read_table('./mapodoufu_2017-08-01.txt',header=None,sep='#')
# mapodoufu2 = pd.read_table('./mapodoufu_2017-07-27.txt',header=None,sep='#')
#
# print sum(mapodoufu1[0]==mapodoufu2[0])
# print sum(mapodoufu1[1]==mapodoufu2[1])
# print sum(mapodoufu1[2]==mapodoufu2[2])
# print sum(mapodoufu1[3]==mapodoufu2[3])
# result=np.sum(np.abs(mapodoufu1[3]-mapodoufu2[3])/mapodoufu2[3])/len(mapodoufu2[3])
# print result