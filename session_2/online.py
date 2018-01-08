# -*- coding: utf-8 -*-
#!/usr/bin/env python
from sklearn import preprocessing
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neighbors
from sklearn.ensemble import RandomForestRegressor
import warnings
import datetime
import time

#online
data = pd.read_csv('traveltime_drop.txt',sep=';',low_memory=False)
seg = pd.read_csv('semifinal_gy_cmp_testing_template_seg2.txt',sep=';',header=None)
seg.columns = ['link_ID','date','time_interval','travel_time']
seg['time_interval_begin'] = pd.to_datetime(seg['time_interval'].map(lambda x: x[1:20]))
seg['w'] = seg['time_interval_begin'].map(lambda x: x.weekday())
# link_ID;date;time_interval;travel_time;w
del seg['time_interval_begin']

data['minut']=pd.Series(data['time_interval'].str.slice(15,17))
data['h']=pd.Series(data['time_interval'].str.slice(12,14))
data['slice']=pd.Series(data['time_interval'].str.slice(15,16))
data['1/travel_time']=1/data['travel_time']
data['1/travel_time2']=1/data['travel_time']/data['travel_time']
seg['h']=pd.Series(seg['time_interval'].str.slice(12,14))
seg['minut']=pd.Series(seg['time_interval'].str.slice(15,17))
seg['slice']=pd.Series(seg['time_interval'].str.slice(15,16))
#缺数据，不加实时
#实时，历史
nowh=data[(data['date']<'2017-07-01') &( (data['h']=='06')| (data['h']=='07') )]
#历史
history=data[(data['date']<'2017-07-01') &( data['h']=='08')]
#模板
seg=seg[(seg['date']>='2017-07-01') &( seg['date']<='2017-07-31')&(seg['h']=='08')]

#link_ID        date time_interval  travel_time
m_m=history.groupby(['link_ID','minut'],as_index=False)['travel_time'].median()
s_m=history.groupby(['link_ID','slice'],as_index=False)['travel_time'].median()
m=history.groupby(['link_ID'],as_index=False)['travel_time'].median()
w_m=history.groupby(['link_ID','w'],as_index=False)['travel_time'].median()

nh_m=nowh.groupby(['link_ID'],as_index=False)['travel_time'].median()
nh_a=nowh.groupby(['link_ID'],as_index=False)['travel_time'].mean()

s1=history.groupby(['link_ID','w'],as_index=False)['1/travel_time'].sum()
s2=history.groupby(['link_ID','w'],as_index=False)['1/travel_time2'].sum()
result=pd.merge(seg, s1, on=['link_ID','w'], how='left')
result.rename(columns={'travel_time': 'true','1/travel_time': 's1'}, inplace=True)
result=pd.merge(result, s2, on=['link_ID','w'], how='left')
result.rename(columns={'1/travel_time2': 's2'}, inplace=True)
result['preloss']=result['s1']/result['s2']

s3=history.groupby(['link_ID'],as_index=False)['1/travel_time'].sum()
s4=history.groupby(['link_ID'],as_index=False)['1/travel_time2'].sum()
result=pd.merge(result, s3, on=['link_ID'], how='left')
result.rename(columns={'1/travel_time': 's3'}, inplace=True)
result=pd.merge(result, s4, on=['link_ID'], how='left')
result.rename(columns={'1/travel_time2': 's4'}, inplace=True)
result['lloss']=result['s3']/result['s4']

s5=history.groupby(['link_ID','slice'],as_index=False)['1/travel_time'].sum()
s6=history.groupby(['link_ID','slice'],as_index=False)['1/travel_time2'].sum()
result=pd.merge(result, s5, on=['link_ID','slice'], how='left')
result.rename(columns={'1/travel_time': 's5'}, inplace=True)
result=pd.merge(result, s6, on=['link_ID','slice'], how='left')
result.rename(columns={'1/travel_time2': 's6'}, inplace=True)
result['sloss']=result['s5']/result['s6']

s7=history.groupby(['link_ID','minut'],as_index=False)['1/travel_time'].sum()
s8=history.groupby(['link_ID','minut'],as_index=False)['1/travel_time2'].sum()
result=pd.merge(result, s7, on=['link_ID','minut'], how='left')
result.rename(columns={'1/travel_time': 's7'}, inplace=True)
result=pd.merge(result, s8, on=['link_ID','minut'], how='left')
result.rename(columns={'1/travel_time2': 's8'}, inplace=True)
result['mloss']=result['s7']/result['s8']

ns1=nowh.groupby(['link_ID'],as_index=False)['1/travel_time'].sum()
ns2=nowh.groupby(['link_ID'],as_index=False)['1/travel_time2'].sum()
result=pd.merge(result, ns1, on=['link_ID'], how='left')
result.rename(columns={'1/travel_time': 'ns1'}, inplace=True)
result=pd.merge(result, ns2, on=['link_ID'], how='left')
result.rename(columns={'1/travel_time2': 'ns2'}, inplace=True)
result['nloss']=result['ns1']/result['ns2']

result=pd.merge(result, m, on=['link_ID'], how='left')
result.rename(columns={'travel_time': 'm'}, inplace=True)
result=pd.merge(result, s_m, on=['link_ID','slice'], how='left')
result.rename(columns={'travel_time': 's_m'}, inplace=True)
result=pd.merge(result, m_m, on=['link_ID','minut'], how='left')
result.rename(columns={'travel_time': 'm_m'}, inplace=True)
result=pd.merge(result, w_m, on=['link_ID','w'], how='left')
result.rename(columns={'travel_time': 'w_m'}, inplace=True)
result=pd.merge(result, nh_m, on=['link_ID'], how='left')
result.rename(columns={'travel_time': 'nh_m'}, inplace=True)
result=pd.merge(result, nh_a, on=['link_ID'], how='left')
result.rename(columns={'travel_time': 'nh_a'}, inplace=True)


result['max1']=(1/result['w_m']+1/result['preloss'])/(1/result['w_m']/result['w_m']+1/result['preloss']/result['preloss'])
result['max2']=(result['preloss']+result['w_m'])/2
result['max3']=(result['preloss']+result['sloss'])/2
result['max4']=(result['preloss']+result['mloss'])/2
result['max5']=(result['sloss']+result['mloss'])/2
result['max6']=(result['sloss']+result['w_m'])/2
result['max7']=(result['mloss']+result['w_m'])/2
result['max8']=(1/result['sloss']+1/result['preloss'])/(1/result['sloss']/result['sloss']+1/result['preloss']/result['preloss'])
result['max9']=(1/result['mloss']+1/result['preloss'])/(1/result['mloss']/result['mloss']+1/result['preloss']/result['preloss'])
result['max10']=(1/result['mloss']+1/result['sloss'])/(1/result['mloss']/result['mloss']+1/result['sloss']/result['sloss'])
result['max11']=(1/result['w_m']+1/result['sloss'])/(1/result['w_m']/result['w_m']+1/result['sloss']/result['sloss'])
result['max12']=(1/result['mloss']+1/result['w_m'])/(1/result['mloss']/result['mloss']+1/result['w_m']/result['w_m'])
result['max13']=(1/result['mloss']+1/result['w_m']+1/result['preloss'])/(1/result['mloss']/result['mloss']+1/result['w_m']/result['w_m']+1/result['preloss']/result['preloss'])


#mape.to_csv('mape8old.csv')
#result.to_csv('online8.csv',index=False)

para=pd.read_csv('para08.txt')
print para.head(5)
del para['1']
temp=pd.DataFrame()
temp=result[['link_ID','preloss','nloss','lloss','sloss','mloss','m' ,'s_m','m_m','w_m','nh_m','nh_a','max1','max2','max3','max4','max5','max6','max7',
             'max8','max9','max10','max11','max12','max13']]
temp=pd.merge(temp,para,on=['link_ID'], how='left')
print temp.head(5)
temp=np.array(temp)

best=np.zeros((len(temp),1))
for i in range(0,len(temp)):
    best[i,0]=temp[i,int(temp[i,25])]
result['traveltime']=pd.DataFrame(best)

result[['link_ID','date','time_interval','traveltime']].to_csv('2017_09_13_08_rule.txt',sep=';',index=False)

print result[['link_ID','date','time_interval','traveltime']].shape
print result[['link_ID','date','time_interval','traveltime']].isnull().sum()
