import numpy as np
import pandas as pd
import sys
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from decimal import Decimal
from tqdm import tqdm
from sklearn import manifold, datasets
import scipy
import altair as alt

np.set_printoptions(threshold=sys.maxsize)

GRB_fit_name = []
trig = []
T90_GRBs = []
T100_GRBs = []
T90_list = []
T100_list = []
temporal_class = []
names = []
grb_jetfit_tsne = []
grb_new = []
FT2_channels = []

TF_before = False
FT_rate = False
single_channel = True
standard = False

summary = pd.read_csv(
    '/Users/alessandraberretta/tSNE_for_GRB_classification/summary_burst_durations.txt', delimiter='|')
trig_time = summary[' Trig_time_met '].values
trig_ID = summary[' Trig_ID '].values
GRB_name = summary['## GRBname '].values
T90_start = summary[' T90_start '].values
T90_stop = summary[' T90_stop '].values
T100_start = summary[' T100_start '].values
T100_stop = summary[' T100_stop '].values

path_GRB = '/Users/alessandraberretta/tSNE_for_GRB_classification/lc_64ms_swift_GRB/'

list_file = [path_GRB +
             file for file in os.listdir(path_GRB) if file.endswith('.lc')]

path_dirs = '/Users/alessandraberretta/'
dirs_list = [path_dirs +
             dir for dir in os.listdir(path_dirs) if dir.endswith('results')]


for idx, elm in enumerate(GRB_name):

    T90 = abs(T90_stop[idx] - T90_start[idx])

    T90_list.append(T90)

reference_point = np.amax(T90_list)/0.064


for idx, elm in enumerate(GRB_name):

    T100 = abs(T100_stop[idx] - T100_start[idx])

    T100_list.append(T100)

reference_point_100 = np.amax(T100_list)/0.064


def get_total_fluence(path_reports):

    reports = [x for x in os.listdir(path_reports) if x.endswith(".txt")]
    total_fluence = []
    names = []
    lines = []
    df_fluence = pd.DataFrame()

    for elm in reports:
        file = open(
            f'/Users/alessandraberretta/tSNE_for_GRB_classification/report/{elm}', "r")
        # index = 0
        # flag = 0
        for line in file:
            # index += 1
            if 'Total' in line and 'Total Fluence' not in line:
                names.append(elm)
                lines.append(line)

        '''
                # if elm == 'GRB130427A_report.txt' or elm == 'GRB200829A_report.txt' or elm == 'GRB080319B_report.txt':
                flag = 1
                total_fluence.append(
                    (line.split(' ')[6], elm))
                break
        if flag == 0:
            # print(total_fluence)
            file.close()
        else:
            file.close()
        '''

    df_fluence['names'] = names
    df_fluence['lines'] = lines

    return total_fluence, df_fluence


total_fluences, df_fluence = get_total_fluence(
    '/Users/alessandraberretta/tSNE_for_GRB_classification/report/')
# df_fluence.to_csv('fluence.csv', index=False, sep='\t')
# fluence_csv = pd.read_csv('fluence.csv', sep='\t')
# lines_fluence = fluence_csv['lines'].values


def nextpower(num, base):
    i = 1
    while i < num:
        i *= base
    return i


pow2 = nextpower(reference_point, 2)

RATE_GRBs = []

GRB_summary = []


def gen_data():

    for elm in tqdm(list_file, desc="gen data"):

        trig_id = elm.split('_')[6][8:-4]

        t90start = T90_start[np.where(trig_ID == float(trig_id))]
        t90stop = T90_stop[np.where(trig_ID == float(trig_id))]
        t100start = T100_start[np.where(trig_ID == float(trig_id))]
        t100stop = T100_stop[np.where(trig_ID == float(trig_id))]
        trigtime = trig_time[np.where(trig_ID == float(trig_id))]
        GRBname = GRB_name[np.where(trig_ID == float(trig_id))]

        '''
        if t90start.size == 0 and t90stop.size == 0 and trigtime.size == 0 and t100start.size == 0 and t100stop.size == 0:
            continue
        '''

        if t90stop.size == 0 and t100stop.size == 0:
            continue

        if trig_id == '310785':
            t90start = 0
            t90stop = 0.3
            print('ok ')
        elif trig_id == '147478':
            t90start = 0
            t90stop = 3
            print('ok ')
        elif trig_id == '284856':
            t90start = 0
            t90stop = 2.0
            print('ok ')
        elif trig_id == '232585':
            t90start = 0
            t90stop = 0.4
            print('ok ')
        elif trig_id == '243690':
            t90start = 0
            t90stop = 0.2
            print('ok ')
        elif trig_id == '299787':
            t90start = 0
            t90stop = 1.8
            print('ok ')
        elif trig_id == '351588':
            t90start = 0
            t90stop = 0.3
            print('ok ')

        lc_64 = fits.open(elm)

        data = lc_64[1].data

        TIME = data['TIME']
        RATE = data['RATE']
        # ERROR = data['ERROR']

        ene_15_25 = []
        ene_25_50 = []
        ene_50_100 = []
        ene_100_350 = []

        fault_grb = []
        '''
        if trig_id == '576738':
            print(t90start+trigtime)
            print(t90stop+trigtime)
            for id, time in enumerate(TIME):
                if time < (t90stop-t90start):
                    print(RATE[id])
        '''

        for idx, r in enumerate(RATE):
            if np.all((r == 0.0)):
                fault_grb.append(False)
            else:
                fault_grb.append(True)
        if not any(fault_grb):
            print('all 0s datafile:', trig_id)
            continue

        for idx, rate in enumerate(RATE):

            if trig_id == '576738' or trig_id == '753445' or trig_id == '352108' or trig_id == '377179' or trig_id == '209351' or trig_id == '449578' or trig_id == '271019' or trig_id == '770958' or trig_id == '426114':
                if TIME[idx] > (t90start+trigtime):
                    ene_15_25.append(rate[0])
                    ene_25_50.append(rate[1])
                    ene_50_100.append(rate[2])
                    ene_100_350.append(rate[3])
            else:
                if TIME[idx] > (t90start+trigtime)-0.1*(np.absolute(t90stop - t90start)) and TIME[idx] < (t90stop+trigtime)+0.1*(np.absolute(t90stop - t90start)):
                    ene_15_25.append(rate[0])
                    ene_25_50.append(rate[1])
                    ene_50_100.append(rate[2])
                    ene_100_350.append(rate[3])

        norm_channels = np.sum(ene_15_25) + np.sum(ene_25_50) + \
            np.sum(ene_50_100) + np.sum(ene_100_350)

        '''
        fluence_csv = pd.read_csv('fluence.csv', sep='\t')
        lines_fluence = fluence_csv['lines'].values
        names_fluence = fluence_csv['names'].values

        for idx, elm in enumerate(lines_fluence):

            name_from_fluence = (names_fluence[idx]).split('_')[
                0] + ' ' + ' ' + ' '
            if GRBname == name_from_fluence:
                if np.float64(elm.split(' ')[0]) and np.float64(elm.split(' ')[1]) and np.float64(elm.split(' ')[2]) and np.float64(elm.split(' ')[3]):
                    norm_ene_15_25_fluence = ene_15_25 / \
                        np.float64(elm.split(' ')[0])
                    norm_ene_25_50_fluence = ene_25_50 / \
                        np.float64(elm.split(' ')[1])
                    norm_ene_50_100_fluence = ene_50_100 / \
                        np.float64(elm.split(' ')[2])
                    norm_ene_100_350_fluence = ene_100_350 / \
                        np.float64(elm.split(' ')[3])


        for idx, elm in enumerate(total_fluences):

            name_from_fluence = (total_fluences[idx][4]).split('_')[
                0] + ' ' + ' ' + ' '

            if GRBname == name_from_fluence:

                print(type(total_fluences[idx][0]))

                if float(total_fluences[idx][0]) == 0:

                    continue

                else:

                    norm_ene_15_25_fluence = ene_15_25 / \
                        np.float64(total_fluences[idx][0])
                    norm_ene_25_50_fluence = ene_25_50 / \
                        np.float64(total_fluences[idx][0])
                    norm_ene_50_100_fluence = ene_50_100 / \
                        np.float64(total_fluences[idx][0])
                    norm_ene_100_350_fluence = ene_100_350 / \
                        np.float64(total_fluences[idx][0])

        list_norm_ene = [norm_ene_15_25_fluence, norm_ene_25_50_fluence,
                         norm_ene_50_100_fluence, norm_ene_100_350_fluence]
        '''
        if norm_channels:
            norm_ene_15_25 = ene_15_25/norm_channels
            norm_ene_25_50 = ene_25_50/norm_channels
            norm_ene_50_100 = ene_50_100/norm_channels
            norm_ene_100_350 = ene_100_350/norm_channels
        else:
            print(trig_id, GRBname[0])
            continue

        list_norm_ene = [norm_ene_15_25, norm_ene_25_50,
                         norm_ene_50_100, norm_ene_100_350]

        padded_channels = []

        for channel in list_norm_ene:
            padded_channels.append(np.pad(
                channel, (0, 2*(pow2)-len(channel)), 'constant'))

        if single_channel:

            if GRBname[0] == 'GRB080503    ':
                T90_single = 0.3
                print('GRB080503    ')
            elif GRBname[0] == 'GRB050724    ':
                T90_single = 3.0
                print('GRB050724    ')
            elif GRBname[0] == 'GRB070714B   ':
                T90_single = 2.0
                print('GRB070714B   ')
            elif GRBname[0] == 'GRB061006   ':
                T90_single = 0.4
                print('GRB061006   ')
            elif GRBname[0] == 'GRB061210   ':
                T90_single = 0.2
                print('GRB061210   ')
            elif GRBname[0] == 'GRB071227   ':
                T90_single = 1.8
                print('GRB071227   ')
            elif GRBname[0] == 'GRB090510   ':
                T90_single = 0.3
                print('GRB090510   ')
            else:
                for idx, trig in enumerate(trig_ID):

                    if trig_id == str(trig):

                        T90_single = abs(T90_stop[idx] - T90_start[idx])

            T90_GRBs.append(T90_single)

            names.append(GRBname[0])

            FT = np.fft.fft(padded_channels[0])/len(padded_channels[0])

            FT2 = np.power(np.absolute(FT), 2)

            yield pd.Series(FT2, index=[f"col{col:02d}" for col in list(range(len(FT2)))])

        if TF_before:

            for chan in padded_channels:

                normalization = np.sum(chan)

                if normalization:

                    for idx, trig in enumerate(trig_ID):

                        if trig_id == str(trig):

                            T90_single = abs(T90_stop[idx] - T90_start[idx])

                            if T90_single < 2:

                                temporal_class.append('short')

                            else:

                                temporal_class.append('long')

                    T90_GRBs.append(T90_single)

                    normalized_chan = chan/normalization

                    FT = np.fft.fft(normalized_chan)/len(normalized_chan)

                    FT2 = np.power(np.absolute(FT), 2)

                    FT2_channels.append(FT2)

            for idx, trig in enumerate(trig_ID):

                if trig_id == str(trig):

                    T90_single = abs(T90_stop[idx] - T90_start[idx])

            T90_GRBs.append(T90_single)

            total_rate = np.concatenate(
                (FT2_channels[0], FT2_channels[1], FT2_channels[2], FT2_channels[3]))

            yield pd.Series(total_rate, index=[f"col{col:02d}" for col in list(range(len(total_rate)))])

        if standard:

            total_rate = np.concatenate(
                (padded_channels[0], padded_channels[1]))

            # total_rate = np.concatenate(
            # (padded_channels[0], padded_channels[1], padded_channels[2], padded_channels[3]))

            if GRBname[0] == 'GRB080503    ':
                T90_single = 0.3
                print('GRB080503    ')
            elif GRBname[0] == 'GRB050724    ':
                T90_single = 3.0
                print('GRB050724    ')
            elif GRBname[0] == 'GRB070714B   ':
                T90_single = 2.0
                print('GRB070714B   ')
            elif GRBname[0] == 'GRB061006   ':
                T90_single = 0.4
                print('GRB061006   ')
            elif GRBname[0] == 'GRB061210   ':
                T90_single = 0.2
                print('GRB061210   ')
            elif GRBname[0] == 'GRB071227   ':
                T90_single = 1.8
                print('GRB071227   ')
            elif GRBname[0] == 'GRB090510   ':
                T90_single = 0.3
                print('GRB090510   ')
            else:
                for idx, trig in enumerate(trig_ID):

                    if trig_id == str(trig):

                        T90_single = abs(T90_stop[idx] - T90_start[idx])

            T90_GRBs.append(T90_single)
            names.append(GRBname[0])

            if FT_rate:

                yield pd.Series(total_rate, index=[f"col{col:02d}" for col in list(range(len(total_rate)))])

            else:

                FT = np.fft.fft(total_rate) / \
                    len(total_rate)

                FT2 = np.power(np.absolute(FT), 2)

                yield pd.Series(FT2, index=[f"col{col:02d}" for col in list(range(len(FT2)))])


df = pd.DataFrame(gen_data())

names_for_comparison = []

for n in names:
    if n[3:-3].endswith(' '):
        names_for_comparison.append(n[3:-4])
    else:
        names_for_comparison.append(n[3:-3])

grb_KN = ['050709A', '050724', '060614A', '070714B',
          '070809A', '080503', '130603B', '150101B', '160821B']
KN = []

for kn in names_for_comparison:
    if kn in grb_KN:
        KN.append('kn')
        print(kn)
    else:
        KN.append('nokn')


df_check = pd.DataFrame()
df_check['name'] = names_for_comparison
df_check.to_csv('check.csv', index=False, sep='\t')

df_comparison = pd.read_csv('/Users/alessandraberretta/JetFit/common_grb.csv')
common = df_comparison['common_grb'].values

tsne = manifold.TSNE(perplexity=30.0, random_state=0, init='pca')
X_embedded = tsne.fit_transform(df)

my_class = []
X_embedded_x = []
X_embedded_y = []

for idx, elm in enumerate(X_embedded):

    X_embedded_x.append(elm[0])
    X_embedded_y.append(elm[1])

    if elm[0] < 5 and elm[0] > -16 and elm[1] < -15 and elm[1] > -45:
        my_class.append('type_S')
    else:
        my_class.append('type_L')

df['X'] = X_embedded_x
df['Y'] = X_embedded_y
df['classification'] = my_class
df['logT90'] = np.log10(T90_GRBs)
df['GRB_name'] = names_for_comparison
df['KN'] = KN

for grb in names_for_comparison:
    if grb.startswith('16') or grb.startswith('17') or grb.startswith('18') or grb.startswith('19') or grb.startswith('20'):
        print(grb)
        grb_new.append('yes')
    else:
        grb_new.append('no')

df['new_grb'] = grb_new

for grb in names_for_comparison:
    if grb in common:
        grb_jetfit_tsne.append('yes')
    else:
        grb_jetfit_tsne.append('no')

df['common_jetfit_tsne'] = grb_jetfit_tsne

df2 = df[df['classification'] == 'type_S']
df3 = df[df['classification'] == 'type_L']
df4 = df[df['common_jetfit_tsne'] == 'yes']
df5 = df[df['common_jetfit_tsne'] == 'no']
df6 = df4[df4['classification'] == 'type_S']
df7 = df[df['KN'] == 'kn']
df8 = df[df['new_grb'] == 'yes']
fig, ax = plt.subplots()

print(df8)

# scatter = plt.scatter(X_embedded_x, X_embedded_y, c=(
# np.log10(T90_GRBs)), cmap='viridis')
# cb = plt.colorbar()
# cb.set_label('log(T90)')

plt.scatter(df4['X'], df4['Y'], c='blue', marker='*',
            alpha=1, s=100, label='Fitted GRBs')
plt.scatter(df5['X'], df5['Y'], c='green',
            alpha=0.3, label='GRBs of BAT sample')
# plt.scatter(df7['X'], df7['Y'], c='red', marker='v',
# alpha=0.8, label='GRBs with KiloNova')
plt.scatter(df8['X'], df8['Y'], c='red', marker='^',
            alpha=0.8, label='new GRBs')
# plt.xlabel('log(T90)')
plt.legend()
plt.show()

'''
# alt.renderers.enable('altair_viewer')
chart = alt.Chart(df).mark_point(size=80, filled=True, opacity=0.6).encode(
    x='X',
    y='Y',
    color=alt.Color('logT90', scale=alt.Scale(scheme='viridis')),
    tooltip=['GRB_name', 'logT90', 'common_jetfit_tsne', 'KN']
).interactive()
chart.save('interactive_scatter_color3.html')
# chart.show()

hist_SL = plt.hist([df2['logT90'], df3['logT90']], label=['S', 'L'],
                   color=['blue', 'red'], alpha=0.3, histtype='stepfilled')
plt.legend()
plt.xlabel('log(T90)')
plt.ylabel('counts')
plt.grid(True)
plt.show()
'''
