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
seg1 = pd.read_csv('padding.txt',sep=';',low_memory=False)
del seg1['travel_time']
data['minut']=pd.Series(data['time_interval'].str.slice(15,17))
data['h']=pd.Series(data['time_interval'].str.slice(12,14))
data['slice']=pd.Series(data['time_interval'].str.slice(15,16))
data['1/travel_time']=1/data['travel_time']
data['1/travel_time2']=1/data['travel_time']/data['travel_time']
seg1['h']=pd.Series(seg1['time_interval'].str.slice(12,14))


#缺数据，不加实时
#实时，历史
nowh=data[(data['date']<'2017-06-01') &((data['h']=='06')|(data['h']=='07'))]
#历史
history=data[(data['date']<'2017-06-01') &( data['h']=='08')]
#模板
seg1=seg1[(seg1['date']>='2017-06-01') &( seg1['date']<='2017-06-30')&( seg1['h']=='08')]
seg=data[(data['date']>='2017-06-01') &( data['date']<='2017-06-30')&( data['h']=='08')]

#link_ID        date time_interval  travel_time
m_m=history.groupby(['link_ID','minut'],as_index=False)['travel_time'].median()
s_m=history.groupby(['link_ID','slice'],as_index=False)['travel_time'].median()
m=history.groupby(['link_ID'],as_index=False)['travel_time'].median()
w_m=history.groupby(['link_ID','w'],as_index=False)['travel_time'].median()

nh_m=nowh.groupby(['link_ID'],as_index=False)['travel_time'].median()
nh_a=nowh.groupby(['link_ID'],as_index=False)['travel_time'].mean()

s1=history.groupby(['link_ID','w'],as_index=False)['1/travel_time'].sum()
s2=history.groupby(['link_ID','w'],as_index=False)['1/travel_time2'].sum()
result=pd.merge(seg1, seg, on=['link_ID','time_interval'], how='left')

result=pd.merge(result, s1, on=['link_ID','w'], how='left')

result.rename(columns={'travel_time': 'true','1/travel_time_y': 's1'}, inplace=True)
result=pd.merge(result, s2, on=['link_ID','w'], how='left')
result.rename(columns={'1/travel_time2_y': 's2'}, inplace=True)
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

ns3=nowh.groupby(['link_ID','w'],as_index=False)['1/travel_time'].sum()
ns4=nowh.groupby(['link_ID','w'],as_index=False)['1/travel_time2'].sum()
result=pd.merge(result, ns3, on=['link_ID','w'], how='left')
result.rename(columns={'1/travel_time': 'ns3'}, inplace=True)
result=pd.merge(result, ns4, on=['link_ID','w'], how='left')
result.rename(columns={'1/travel_time2': 'ns4'}, inplace=True)
result['nwloss']=result['ns3']/result['ns4']


ns5=nowh.groupby(['link_ID','slice'],as_index=False)['1/travel_time'].sum()
ns6=nowh.groupby(['link_ID','slice'],as_index=False)['1/travel_time2'].sum()
result=pd.merge(result, ns5, on=['link_ID','slice'], how='left')
result.rename(columns={'1/travel_time': 'ns5'}, inplace=True)
result=pd.merge(result, ns6, on=['link_ID','slice'], how='left')
result.rename(columns={'1/travel_time2': 'ns6'}, inplace=True)
result['nsloss']=result['ns5']/result['ns6']

ns7=nowh.groupby(['link_ID','minut'],as_index=False)['1/travel_time'].sum()
ns8=nowh.groupby(['link_ID','minut'],as_index=False)['1/travel_time2'].sum()
result=pd.merge(result, ns7, on=['link_ID','minut'], how='left')
result.rename(columns={'1/travel_time': 'ns7'}, inplace=True)
result=pd.merge(result, ns8, on=['link_ID','minut'], how='left')
result.rename(columns={'1/travel_time2': 'ns8'}, inplace=True)
result['nmloss']=result['ns7']/result['ns8']




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

result['max14']=(result['preloss']+result['nwloss'])/2
result['max15']=(result['preloss']+result['nsloss'])/2
result['max16']=(result['preloss']+result['nmloss'])/2
result['max17']=(1/result['preloss']+1/result['nwloss'])/(1/result['preloss']/result['preloss']+1/result['nwloss']/result['nwloss'])
result['max18']=(1/result['preloss']+1/result['nsloss'])/(1/result['preloss']/result['preloss']+1/result['nsloss']/result['nsloss'])
result['max19']=(1/result['preloss']+1/result['nmloss'])/(1/result['preloss']/result['preloss']+1/result['nmloss']/result['nmloss'])


#link_ID    date  time_interval   preloss     nloss    m  s_m   m_m  w_m  nh_m      nh_a

result['mapepreloss']=abs(result['preloss']-result['true'])/result['true']
result['mapenloss']=abs(result['nloss']-result['true'])/result['true']
result['lloss']=abs(result['lloss']-result['true'])/result['true']
result['sloss']=abs(result['sloss']-result['true'])/result['true']
result['mloss']=abs(result['mloss']-result['true'])/result['true']
result['mapem']=abs(result['m']-result['true'])/result['true']
result['mapes_m']=abs(result['s_m']-result['true'])/result['true']
result['mapem_m']=abs(result['m_m']-result['true'])/result['true']
result['mapew_m']=abs(result['w_m']-result['true'])/result['true']
result['mapenh_m']=abs(result['nh_m']-result['true'])/result['true']
result['mapenh_a']=abs(result['nh_a']-result['true'])/result['true']
result['max1']=abs(result['max1']-result['true'])/result['true']
result['max2']=abs(result['max2']-result['true'])/result['true']

result['max3']=abs(result['max3']-result['true'])/result['true']
result['max4']=abs(result['max4']-result['true'])/result['true']
result['max5']=abs(result['max5']-result['true'])/result['true']
result['max6']=abs(result['max6']-result['true'])/result['true']
result['max7']=abs(result['max7']-result['true'])/result['true']
result['max8']=abs(result['max8']-result['true'])/result['true']
result['max9']=abs(result['max9']-result['true'])/result['true']
result['max10']=abs(result['max10']-result['true'])/result['true']
result['max11']=abs(result['max11']-result['true'])/result['true']
result['max12']=abs(result['max12']-result['true'])/result['true']
result['max13']=abs(result['max13']-result['true'])/result['true']
result['max14']=abs(result['max14']-result['true'])/result['true']
result['max15']=abs(result['max15']-result['true'])/result['true']
result['max16']=abs(result['max16']-result['true'])/result['true']
result['max17']=abs(result['max17']-result['true'])/result['true']
result['max18']=abs(result['max18']-result['true'])/result['true']
result['max19']=abs(result['max19']-result['true'])/result['true']

result['nwloss']=abs(result['nwloss']-result['true'])/result['true']
result['nsloss']=abs(result['nsloss']-result['true'])/result['true']
result['nmloss']=abs(result['nmloss']-result['true'])/result['true']

mapepreloss=result.groupby(['link_ID'],as_index=False)['mapepreloss'].mean()
mapenloss=result.groupby(['link_ID'],as_index=False)['mapenloss'].mean()
lloss=result.groupby(['link_ID'],as_index=False)['lloss'].mean()
sloss=result.groupby(['link_ID'],as_index=False)['sloss'].mean()
mloss=result.groupby(['link_ID'],as_index=False)['mloss'].mean()
mapem=result.groupby(['link_ID'],as_index=False)['mapem'].mean()
mapes_m=result.groupby(['link_ID'],as_index=False)['mapes_m'].mean()
mapem_m=result.groupby(['link_ID'],as_index=False)['mapem_m'].mean()
mapew_m=result.groupby(['link_ID'],as_index=False)['mapew_m'].mean()
mapenh_m=result.groupby(['link_ID'],as_index=False)['mapenh_m'].mean()
mapenh_a=result.groupby(['link_ID'],as_index=False)['mapenh_a'].mean()
max1=result.groupby(['link_ID'],as_index=False)['max1'].mean()
max2=result.groupby(['link_ID'],as_index=False)['max2'].mean()
max3=result.groupby(['link_ID'],as_index=False)['max3'].mean()
max4=result.groupby(['link_ID'],as_index=False)['max4'].mean()
max5=result.groupby(['link_ID'],as_index=False)['max5'].mean()
max6=result.groupby(['link_ID'],as_index=False)['max6'].mean()
max7=result.groupby(['link_ID'],as_index=False)['max7'].mean()
max8=result.groupby(['link_ID'],as_index=False)['max8'].mean()
max9=result.groupby(['link_ID'],as_index=False)['max9'].mean()
max10=result.groupby(['link_ID'],as_index=False)['max10'].mean()
max11=result.groupby(['link_ID'],as_index=False)['max11'].mean()
max12=result.groupby(['link_ID'],as_index=False)['max12'].mean()
max13=result.groupby(['link_ID'],as_index=False)['max13'].mean()

max14=result.groupby(['link_ID'],as_index=False)['max14'].mean()
max15=result.groupby(['link_ID'],as_index=False)['max15'].mean()
max16=result.groupby(['link_ID'],as_index=False)['max16'].mean()
max17=result.groupby(['link_ID'],as_index=False)['max17'].mean()
max18=result.groupby(['link_ID'],as_index=False)['max18'].mean()
max19=result.groupby(['link_ID'],as_index=False)['max19'].mean()

nwloss=result.groupby(['link_ID'],as_index=False)['nwloss'].mean()
nsloss=result.groupby(['link_ID'],as_index=False)['nsloss'].mean()
nmloss=result.groupby(['link_ID'],as_index=False)['nmloss'].mean()



temp=pd.DataFrame()
temp=pd.concat([mapepreloss['mapepreloss'],mapenloss['mapenloss'],lloss['lloss'],sloss['sloss'],mloss['mloss'],
                mapem['mapem'] ,mapes_m['mapes_m'],mapem_m['mapem_m'],mapew_m['mapew_m'],mapenh_m['mapenh_m'],mapenh_a['mapenh_a'],max1['max1'] ,max2['max2'],
                max3['max3'], max4['max4'],max5['max5'] ,max6['max6'],max7['max7'] ,max8['max8'],max9['max9'] ,max10['max10'],max11['max11'] ,max12['max12'],max13['max13'] ],axis=1)
print temp.head(5)
mape=np.array(temp)
best=np.zeros((len(temp),2))
for i in range(0,len(temp)):
    best[i,0]=1
    best[i,1]=mape[i,0] #temp mape
    if mape[i,1]<best[i,1]:
        best[i,0]=2
        best[i, 1] = mape[i, 1]
    if mape[i,2]<best[i,1]:
        best[i,0]=3
        best[i, 1] = mape[i, 2]
    if mape[i,3]<best[i,1]:
        best[i,0]=4
        best[i, 1] = mape[i, 3]
    if mape[i,4]<best[i,1]:
        best[i,0]=5
        best[i, 1] = mape[i, 4]
    if mape[i,5]<best[i,1]:
        best[i,0]=6
        best[i, 1] = mape[i, 5]
    if mape[i,6]<best[i,1]:
        best[i,0]=7
        best[i, 1] = mape[i, 6]
    if mape[i,7]<best[i,1]:
        best[i,0]=8
        best[i, 1] = mape[i, 7]
    if mape[i,8]<best[i,1]:
        best[i,0]=9
        best[i, 1] = mape[i, 8]
    if mape[i,9]<best[i,1]:
        best[i,0]=10
        best[i, 1] = mape[i, 9]
    if mape[i,10]<best[i,1]:
        best[i,0]=11
        best[i, 1] = mape[i, 10]
    if mape[i,11]<best[i,1]:
        best[i,0]=12
        best[i, 1] = mape[i, 11]
    if mape[i,12]<best[i,1]:
        best[i,0]=13
        best[i, 1] = mape[i, 12]
    if mape[i,13]<best[i,1]:
        best[i,0]=14
        best[i, 1] = mape[i, 13]
    if mape[i,14]<best[i,1]:
        best[i,0]=15
        best[i, 1] = mape[i, 14]
    if mape[i,15]<best[i,1]:
        best[i,0]=16
        best[i, 1] = mape[i, 15]
    if mape[i,16]<best[i,1]:
        best[i,0]=17
        best[i, 1] = mape[i, 16]
    if mape[i,17]<best[i,1]:
        best[i,0]=18
        best[i, 1] = mape[i, 17]
    if mape[i,18]<best[i,1]:
        best[i,0]=19
        best[i, 1] = mape[i, 18]
    if mape[i,19]<best[i,1]:
        best[i,0]=20
        best[i, 1] = mape[i, 19]
    if mape[i,20]<best[i,1]:
        best[i,0]=21
        best[i, 1] = mape[i, 20]
    if mape[i,21]<best[i,1]:
        best[i,0]=22
        best[i, 1] = mape[i, 21]
    if mape[i,22]<best[i,1]:
        best[i,0]=23
        best[i, 1] = mape[i, 22]
    if mape[i,23]<best[i,1]:
        best[i,0]=24
        best[i, 1] = mape[i, 23]

testmape=best[:,1]
testmape=testmape[testmape>0]
print testmape
print testmape.mean()
para=pd.concat([mapepreloss['link_ID'],pd.DataFrame(best.astype(int))],axis=1)

para.to_csv('para08.txt',index=False)

# 15 0.273884567108
# 18 0.269130923414
# 08 0.287181842916