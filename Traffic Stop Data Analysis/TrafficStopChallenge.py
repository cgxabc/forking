#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 14:02:13 2018

@author: apple
"""

import pandas as pd 
import numpy as np

mt_data = pd.read_csv('MT_cleaned.csv')
vt_data = pd.read_csv('VT_cleaned.csv')

###Q1
mt_male_data = mt_data[mt_data['driver_gender'] == 'M']
mt_female_data = mt_data[mt_data['driver_gender'] == 'F']
proportion = float(mt_male_data.shape[0])/len(mt_data)
proportion    ##0.668128763111

###Q2
mt_female_data['is_arrested'].value_counts()   
mt_male_data['is_arrested'].value_counts() 

female_not_arrested = 43291
female_arrested = 1559
male_not_arrested = 86870
male_arrested = 3457

from scipy.stats import chi2_contingency
obs = np.array([[female_arrested, female_not_arrested],[male_arrested, male_not_arrested]])
chi2_contingency(obs)[0]


####Q3
mt_data['violation'].fillna('Unknown',inplace = True)
mt_prop_dui = mt_data.loc[mt_data['violation'].str.contains('DUI')].shape[0]/float(mt_data.shape[0])
vt_data['violation'].fillna('Unknown',inplace = True)
vt_prop_dui = vt_data.loc[vt_data['violation'].str.contains('DUI')].shape[0]/float(vt_data.shape[0])
mt_prop_dui-vt_prop_dui

####Q4
mt_arrested_prop = len(mt_data[mt_data['is_arrested'] == True])/float(mt_data.shape[0])
mt_arrested_outofstate_prop = len(mt_data[(mt_data['out_of_state'] == True) & (mt_data['is_arrested'] == True)])/float(len(mt_data[mt_data['out_of_state'] == True]))
mt_arrested_outofstate_prop - mt_arrested_prop

####Q5
mt_prop_speeding = mt_data.loc[mt_data['violation'].str.contains('Speeding')].shape[0]/float(mt_data.shape[0])
mt_prop_speeding

####Q6, Q7
mt_data['stop_date'].fillna('2009-01-01',inplace = True)
mt_data.loc[135194, 'stop_date'] = '2010-11-27'  
mt_data['Year'] = mt_data.stop_date.apply(lambda x: int(str(x).split('-')[0]))
mt_data_vehicle = mt_data.dropna(subset = ['vehicle_year'])
mt_data_vehicle['vehicle_year'] = mt_data_vehicle.vehicle_year.apply(pd.to_numeric, errors = 'coerce')
mt_data_vehicle = mt_data_vehicle.dropna(subset = ['vehicle_year'])
mt_data_vehicle.groupby(['Year'])['vehicle_year'].mean()
year_table = pd.DataFrame({'Year': [2009.0, 2010.0], 'Ave Vehicle Year': [2000.980215, 2001.521045]})

from sklearn import linear_model
from sklearn.linear_model import LinearRegression

clf = LinearRegression()
features = ['Year']
clf.fit(year_table[features], year_table['Ave Vehicle Year'])
intercept = clf.intercept_
coef = clf.coef_
pd.DataFrame({'name': ['intercept']+features, 'value': [intercept]+list(coef)})
ave_vehicle_year_2020 = intercept + coef*2020
ave_vehicle_year_2020[0]

from scipy import stats
stats.linregress(year_table['Year'].values, year_table['Ave Vehicle Year'].values)[3]

####Q8
mt_data = mt_data.dropna(subset=['stop_time'])
vt_data = vt_data.dropna(subset=['stop_time'])
mt_data['Hour'] = mt_data.stop_time.apply(lambda x: x.split(':')[0])
vt_data['Hour'] = vt_data.stop_time.apply(lambda x: x.split(':')[0])
combined_hour = pd.concat([mt_data['Hour'], vt_data['Hour']])
combined_hour.value_counts()[0:].values[0]-combined_hour.value_counts()[23:].values[0]

####Q9
mt_location = mt_data.dropna(subset = ['lat','lon'])
mt_location = mt_location[(mt_location['lat'] < 50) & (mt_location['lat'] > 40)]
mt_location = mt_location[mt_location['lon'] < -80]
import math
sorted(mt_location.groupby('county_name').apply(lambda x: math.pi*x['lat'].std()*x['lon'].std()),reverse=True)[0]














