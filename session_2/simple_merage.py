#coding:utf-8

import pandas as pd


# a_p = pd.read_table('./mapodoufu_2017-09-15_a_lgb.txt',sep="#")
# print a_p[['link_ID','date','time_interval','travel_time']].shape
# print a_p[['link_ID','date','time_interval','travel_time']].isnull().sum()

# b_p = pd.read_table('./mapodoufu_2017-09-15_b_lgb.txt',sep="#")
# print b_p[['link_ID','date','time_interval','travel_time']].shape
# print b_p[['link_ID','date','time_interval','travel_time']].isnull().sum()

# c_p = pd.read_table('./mapodoufu_2017-09-15_c_lgb.txt',sep="#")
# print c_p[['link_ID','date','time_interval','travel_time']].shape
# print c_p[['link_ID','date','time_interval','travel_time']].isnull().sum()

# a = pd.read_table('./mapodoufu_2017-09-15_a_xgb.txt',sep="#")
# print a[['link_ID','date','time_interval','travel_time']].shape
# print a[['link_ID','date','time_interval','travel_time']].isnull().sum()

# b = pd.read_table('./mapodoufu_2017-09-15_b_xgb.txt',sep="#")
# print b[['link_ID','date','time_interval','travel_time']].shape
# print b[['link_ID','date','time_interval','travel_time']].isnull().sum()

# c = pd.read_table('./mapodoufu_2017-09-15_c_xgb.txt',sep="#")
# print c[['link_ID','date','time_interval','travel_time']].shape
# print c[['link_ID','date','time_interval','travel_time']].isnull().sum()


# new_a = pd.merge(a,a_p,on=['link_ID','date','time_interval'],how='left')
# new_b = pd.merge(b,b_p,on=['link_ID','date','time_interval'],how='left')
# new_c = pd.merge(c,c_p,on=['link_ID','date','time_interval'],how='left')

# new_a['travel_time'] = 0.5 * new_a['travel_time_x'] + 0.5 * new_a['travel_time_y']
# new_b['travel_time'] = 0.5 * new_b['travel_time_x'] + 0.5 * new_b['travel_time_y']
# new_c['travel_time'] = 0.5 * new_c['travel_time_x'] + 0.5 * new_c['travel_time_y']

# new_a[['link_ID','date','time_interval','travel_time']].to_csv('./mapoudoufu_2017_09_15_a_xgb_lgb.txt',sep='#',index=False)
# print new_a[['link_ID','date','time_interval','travel_time']].shape
# print new_a[['link_ID','date','time_interval','travel_time']].isnull().sum()
# print new_a
# new_b[['link_ID','date','time_interval','travel_time']].to_csv('./mapoudoufu_2017_09_15_b_xgb_lgb.txt',sep='#',index=False)
# print new_b[['link_ID','date','time_interval','travel_time']].shape
# print new_b[['link_ID','date','time_interval','travel_time']].isnull().sum()
# print new_b
# new_c[['link_ID','date','time_interval','travel_time']].to_csv('./mapoudoufu_2017_09_15_c_xgb_lgb.txt',sep='#',index=False)
# print new_c[['link_ID','date','time_interval','travel_time']].shape
# print new_c[['link_ID','date','time_interval','travel_time']].isnull().sum()
# print new_c

import numpy as np

a_p = pd.read_table('./mapoudoufu_2017_09_15_a_xgb_lgb.txt',sep="#")
b_p = pd.read_table('./mapoudoufu_2017_09_15_b_xgb_lgb.txt',sep="#")
c_p = pd.read_table('./mapoudoufu_2017_09_15_c_xgb_lgb.txt',sep="#")

a = pd.read_table('./frame08final.txt',sep=";")
b = pd.read_table('./frame15final.txt',sep=";")
c = pd.read_table('./frame18final.txt',sep=";")
print a[['link_ID','date','time_interval','travel_time']].shape
print a[['link_ID','date','time_interval','travel_time']].isnull().sum()
print b[['link_ID','date','time_interval','travel_time']].shape
print b[['link_ID','date','time_interval','travel_time']].isnull().sum()
print c[['link_ID','date','time_interval','travel_time']].shape
print c[['link_ID','date','time_interval','travel_time']].isnull().sum()

new_a = pd.merge(a,a_p,on=['link_ID','date','time_interval'],how='left')
new_b = pd.merge(b,b_p,on=['link_ID','date','time_interval'],how='left')
new_c = pd.merge(c,c_p,on=['link_ID','date','time_interval'],how='left')

new_a['travel_time'] = 0.5 * new_a['travel_time_x'] + 0.5 * new_a['travel_time_y']
new_b['travel_time'] = 0.5 * new_b['travel_time_x'] + 0.5 * new_b['travel_time_y']
new_c['travel_time'] = 0.5 * new_c['travel_time_x'] + 0.5 * new_c['travel_time_y']

new_c['travel_time'] = np.round(new_c['travel_time'],6)
new_b['travel_time'] = np.round(new_b['travel_time'],6)
new_a['travel_time'] = np.round(new_a['travel_time'],6)

new_a[['link_ID','date','time_interval','travel_time']].to_csv('./mapoudoufu_2017_09_15_a_sub.txt',sep='#',index=False)
print new_a[['link_ID','date','time_interval','travel_time']].shape
print new_a[['link_ID','date','time_interval','travel_time']].isnull().sum()
print new_a.head()
new_b[['link_ID','date','time_interval','travel_time']].to_csv('./mapoudoufu_2017_09_15_b_sub.txt',sep='#',index=False)
print new_b[['link_ID','date','time_interval','travel_time']].shape
print new_b[['link_ID','date','time_interval','travel_time']].isnull().sum()
print new_b.head()
new_c[['link_ID','date','time_interval','travel_time']].to_csv('./mapoudoufu_2017_09_15_c_sub.txt',sep='#',index=False)
print new_c[['link_ID','date','time_interval','travel_time']].shape
print new_c[['link_ID','date','time_interval','travel_time']].isnull().sum()
print new_c.head()