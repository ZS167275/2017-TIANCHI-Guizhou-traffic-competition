#coding:utf-8
import pandas as pd
import numpy as np

# 大赛提供132条link的静态信息，以及这些link之间的上下游拓扑结构。
# 同时，大赛提供2016年3月至2016年5月每条link每天的旅行时间，以及2016年6月早上[6:00- 8:00)每条link的平均旅行时间。
########################################################################################################################
# link_ID 每条路段的唯一标识
# length link长度
# width link宽度
# link_class 路段等级
# gy_contest_link_info = pd.read_table('./gy_contest_link_info.txt',sep=';')
# print gy_contest_link_info
########################################################################################################################
#link_ID 每条路段的唯一标识
#in_links link的直接上游link
#out_links link的直接下游link
# gy_contest_link_top = pd.read_table('./gy_contest_link_top.txt',sep=';')
# print gy_contest_link_top['in_links'].map(lambda x:str(x).split('#'))
########################################################################################################################
#link_ID 每条路段的唯一标识
#date 日期
#time_interval 时间段
#travel_time 旅行时间
########################################################################################################################
from scipy.stats import mode
#
def mode_function(df):
    counts = mode(df)
    return counts[0][0]

# 1.1 这个部分是读取新的数据格式的文件，然后把时间格式划分一下/
gy_contest_link_traveltime_training_data = pd.read_table(u'./[新-训练集]gy_contest_traveltime_training_data_second.txt',sep=';')
gy_contest_link_traveltime_training_data.columns = ['link_ID', 'date', 'time_interval', 'travel_time']
sub = gy_contest_link_traveltime_training_data.sort_values(['link_ID','time_interval'])
Select_data = sub
# ########## 转换time_interval格式为begin/end ##########
Select_data['time_interval_begin'] = pd.to_datetime(Select_data.time_interval.map(lambda x : x[1:20]))
# Select_data['time_interval_end'] = pd.to_datetime(Select_data.time_interval.map(lambda x : x[21:-1]))
Select_data['time_interval_month'] = Select_data['time_interval_begin'].map(lambda x: x.strftime('%m'))
Select_data['time_interval_hour'] = Select_data['time_interval_begin'].map(lambda x : x.strftime('%H'))
Select_data['time_interval_minutes'] = Select_data['time_interval_begin'].map(lambda x : x.strftime('%M'))
Select_data['time_interval_day'] = Select_data['time_interval_begin'].map(lambda x : x.day)
print Select_data.head()
这个部分可以修改 04 05 代表读取这个时段的数据
Select_data = Select_data[(Select_data['time_interval_month']=='04')]
print Select_data.date.unique()
Select_data.to_csv('./ruler_5_feat_data.csv',index=False)


# 1.2 这个部分是统计中位数和众数
Select_data = pd.read_csv('./ruler_4_feat_data.csv')
print Select_data.head()
# tmp = Select_data[Select_data['time_interval_month']==5]
Select_data['link_ID'] = Select_data['link_ID'].astype('str')
tmp_count = Select_data.groupby(['link_ID','time_interval_minutes'])['travel_time'].agg([('mean_',np.mean),('median_',np.median),('std_',np.std),('mode_',mode_function),('max_',np.max)]).reset_index()
tmp_count.to_csv('./count_4_feat.csv',index=False)


# 这个部分是测试线下
train_org_feat = pd.read_csv('./count_4_feat.csv')
train_org_feat = train_org_feat.fillna(0)
# count_train = train_org_feat[train_org_feat['time_interval_month']==4]
count_train = train_org_feat[['link_ID','time_interval_minutes','mean_','median_','mode_','std_','max_']]
count_train['mode_median_'] = 0.50 * count_train['median_'] + 0.50 * count_train['mode_']
count_train.to_csv('./x.csv',index=False)

# 这个部分是读取上一步的数据，
count_train = pd.read_csv('./x.csv')
count_train['link_ID'] = count_train['link_ID'].astype('str')
print count_train



# 这个步骤需要先使用上面 1.1 1.2 更改为5月份 目的是提取 路段 - 分钟
test = pd.read_csv('./ruler_5_feat_data.csv')
test['link_ID'] = test['link_ID'].astype('str')
print test
train = pd.merge(test,count_train,on=['link_ID','time_interval_minutes'],how='left')
result=np.sum(np.abs(train['mode_median_']-train['travel_time'].values)/train['travel_time'].values)/len(train['travel_time'].values)
result1=np.sum(np.abs(train['mode_']-train['travel_time'].values)/train['travel_time'].values)/len(train['travel_time'].values)
result2=np.sum(np.abs(train['median_']-train['travel_time'].values)/train['travel_time'].values)/len(train['travel_time'].values)
result3=np.sum(np.abs(train['mean_']-train['travel_time'].values)/train['travel_time'].values)/len(train['travel_time'].values)
print result,result1,result2,result3
print train



# 这个部分是读取提交的数据的格式

sub_demo = pd.read_table(u'[新-答案模板]gy_contest_result_template.txt',header=None,sep='#')

sub_demo.columns = ['link_ID','date','time_interval','travel_time']
sub_demo = sub_demo.sort_values(['link_ID','time_interval']).reset_index()
del sub_demo['index']
del sub_demo['travel_time']

sub_demo['time_interval_begin'] = pd.to_datetime(sub_demo.time_interval.map(lambda x : x[1:20]))
sub_demo['time_interval_minutes'] = sub_demo['time_interval_begin'].map(lambda x : x.strftime('%M'))
sub_demo['time_interval_minutes'] = sub_demo['time_interval_minutes'].astype(int)
print sub_demo
sub_demo['link_ID'] = sub_demo['link_ID'].astype('str')
del sub_demo['time_interval_begin']

sub_demo.to_csv('./tmp_sub.csv',index=False)



# 这个部分是进行拼接 拼接6月和5月的统计数据
tmp_sub = pd.read_csv('./tmp_sub.csv')
tmp_sub = tmp_sub.drop_duplicates(['link_ID','time_interval'])
print tmp_sub[tmp_sub['link_ID']=='4377906289813600514']
tmp_count = pd.read_csv('./count_5_feat.csv')
print tmp_count[tmp_count['link_ID']=='4377906289813600514']
tmp_count['link_ID'] = tmp_count['link_ID'].astype('str')
tmp_count['time_interval_minutes'] = tmp_count['time_interval_minutes'].astype(int)

sub_demo= pd.merge(tmp_sub,tmp_count,on=['link_ID','time_interval_minutes'],how='left')
print sub_demo


sub_demo['t'] = 0.50 * sub_demo['median_'] + 0.50 * sub_demo['mode_'] + 0.001 * sub_demo['max_'] - 0.001
sub_demo = sub_demo[['link_ID','date','time_interval','median_','mode_','t','std_','mean_']]
sub_demo[['link_ID','date','time_interval','t']].to_csv('./siyueshinidehuangyan_2017-08-11.txt',sep='#',index=False,header=False)
print sub_demo[['link_ID','date','time_interval','t']].shape
print sub_demo[['link_ID','date','time_interval','t']].isnull().sum()
