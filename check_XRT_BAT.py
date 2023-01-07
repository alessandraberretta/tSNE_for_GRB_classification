from unicodedata import name
import numpy as np
import pandas as pd
import os
import sys


path_fitted_XRT = '/Users/alessandraberretta/'
listGRB_BAT_XRT_tSNE = '/Users/alessandraberretta/tSNE_for_GRB_classification/names_for_jetfit.csv'
dead_GRB = '/Users/alessandraberretta/tSNE_for_GRB_classification/check.csv'
df_dead = pd.read_csv(dead_GRB)
names = df_dead['name'].values
names_normal = []
names_normal_2 = []

for n in names:
    names_normal.append(n[2:-5])

for n2 in names_normal:
    if n2.endswith(' '):
        names_normal_2.append(n2[3:-1])
    else:
        names_normal_2.append(n2[3:])

dirs_list = [path_fitted_XRT +
             dir for dir in os.listdir(path_fitted_XRT) if dir.endswith('results')]
GRB_names = []
for elm in dirs_list:
    for file in [elm + '/' + file for file in os.listdir(elm) if file.startswith('lc_')]:
        GRB_names.append(file.split('_')[5])

df = pd.read_csv(listGRB_BAT_XRT_tSNE)
col_names = df['names'].values

present = []
not_present = []

for elm in GRB_names:
    if elm in col_names:
        present.append(elm)
        '''
        if elm.startswith('13'):
            print(elm)
        '''
    else:
        not_present.append(elm)
        print(elm)
        '''
        if elm.startswith('17'):
            print(elm)
        '''

for elm2 in names_normal_2:
    if elm2 in GRB_names:
        print('GRB eliminated after the normalization:', elm2)
