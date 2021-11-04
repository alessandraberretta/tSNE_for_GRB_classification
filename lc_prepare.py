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
FT2_channels = []
trigID_str = []
strange_trigID = []

TF_before = False
FT_rate = False
single_channel = True
standard = False

summary = pd.read_csv(
    '/Users/alessandraberretta/tSNE_for_GRB_classification/summary_burst_durations.txt', delimiter='|')
trig_time = summary[' Trig_time_met '].values
trig_ID = summary[' Trig_ID '].values
for trig in trig_ID:
    trigID_str.append(trig)
    if len(str(trig)) == 8:
        datafile = '/Users/alessandraberretta/tSNE_for_GRB_classification/lc_64ms_swift_GRB/sw000' + \
            str(trig) + 'b_4chan_64ms.lc'
        strange_trigID.append((str(trig), datafile))
    if len(str(trig)) == 9:
        datafile = '/Users/alessandraberretta/tSNE_for_GRB_classification/lc_64ms_swift_GRB/sw00' + \
            str(trig) + 'b_4chan_64ms.lc'
        strange_trigID.append((str(trig), datafile))

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

    for elm in reports:

        file = open(
            f'/Users/alessandraberretta/tSNE_for_GRB_classification/report/{elm}', "r")
        index = 0
        flag = 0

        for line in file:

            index += 1

            if 'Total' in line and 'Total Fluence' not in line:

                flag = 1
                total_fluence.append((line.split(' ')[6], elm))
                break

        if flag == 0:
            file.close()
        else:
            file.close()
    return total_fluence


def nextpower(num, base):
    i = 1
    while i < num:
        i *= base
    return i


pow2 = nextpower(reference_point, 2)

RATE_GRBs = []

GRB_summary = []

'''
for elm in tqdm(list_file):
    trig_id = elm.split('_')[6][8:-4]
    for id, BAT_trig in enumerate(trig_ID):
        if trig_id in trig_ID:
            t90start = T90_start[id]
            t90stop = T90_stop[id]
            t100start = T100_start[id]
            t100stop = T100_stop[id]
            trigtime = trig_time[id]
            GRBname = GRB_name[id]
            GRB_summary.append((GRBname, trig_ID[id], trigtime, t90start,
                                t90stop, t100start, t100stop))
print(len(GRB_summary))
'''


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
        # trig_id = elm.split('_')[7][8:-4]
        for i, el in enumerate(trig_ID):
            if trig_id in trig_ID:
                t90start = T90_start[i]
                t90stop = T90_stop[i]
                t100start = T100_start[i]
                t100stop = T100_stop[i]
                trigtime = trig_time[i]
                GRBname = GRB_name[i]
        '''

        '''
        if t90start.size == 0 and t90stop.size == 0 and trigtime.size == 0 and t100start.size == 0 and t100stop.size == 0:
            continue
        '''

        if t90stop.size == 0 and t100stop.size == 0:
            continue

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

                if TIME[idx] < (t90stop+trigtime) and TIME[idx] > (t90start+trigtime):
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
        norm_fluence = (t90stop - t90start)*(np.sum(ene_15_25) +
                                             np.sum(ene_25_50) + np.sum(ene_50_100) + np.sum(ene_100_350))
        '''

        '''
        total_fluences = get_total_fluence(
            '/Users/alessandraberretta/tSNE_for_GRB_classification/report/')

        for idx, elm in enumerate(total_fluences):

            name_from_fluence = (total_fluences[idx][1]).split('_')[
                0] + ' ' + ' ' + ' '

            if GRBname == name_from_fluence:

                if total_fluences[idx][0] == '':

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

        list_norm_ene_fluence = [norm_ene_15_25_fluence, norm_ene_25_50_fluence,
                                 norm_ene_50_100_fluence, norm_ene_100_350_fluence]

        '''
        if norm_channels:
            norm_ene_15_25 = ene_15_25/norm_channels
            norm_ene_25_50 = ene_25_50/norm_channels
            norm_ene_50_100 = ene_50_100/norm_channels
            norm_ene_100_350 = ene_100_350/norm_channels
        else:
            print(trig_id)
            continue

        list_norm_ene = [norm_ene_15_25, norm_ene_25_50,
                         norm_ene_50_100, norm_ene_100_350]

        padded_channels = []

        for channel in list_norm_ene:
            padded_channels.append(np.pad(
                channel, (0, (pow2)-len(channel)), 'constant'))

        if single_channel:

            for idx, trig in enumerate(trig_ID):

                if trig_id == str(trig):

                    T90_single = abs(T90_stop[idx] - T90_start[idx])

            T90_GRBs.append(T90_single)

            names.append(GRBname[0])
            # names.append(elm[0])

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
                (padded_channels[0], padded_channels[1], padded_channels[2], padded_channels[3]))

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

df_check = pd.DataFrame()
df_check['name'] = names_for_comparison
df_check.to_csv('check.csv', index=False, sep='\t')

df_comparison = pd.read_csv('/Users/alessandraberretta/JetFit/common_grb.csv')
common = df_comparison['common_grb'].values

# df.to_csv('fft_GRBs.txt', index=False, sep=' ')
# df.to_parquet("fft_GRBs.parquet")


tsne = manifold.TSNE(perplexity=30.0, random_state=1)
X_embedded = tsne.fit_transform(df)

my_class = []
X_embedded_x = []
X_embedded_y = []

for idx, elm in enumerate(X_embedded):

    X_embedded_x.append(elm[0])
    X_embedded_y.append(elm[1])

    if elm[0] < 7.5 and elm[0] > -10 and elm[1] < -15 and elm[1] > -45:
        my_class.append('type_S')
    else:
        my_class.append('type_L')

df['X'] = X_embedded_x
df['Y'] = X_embedded_y
df['classification'] = my_class
df['logT90'] = np.log10(T90_GRBs)
df['GRB_name'] = names_for_comparison

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
print(df6)
fig, ax = plt.subplots()

# scatter = plt.scatter(X_embedded_x, X_embedded_y, c=(
# np.log10(T90_GRBs)), cmap='viridis')
# cb = plt.colorbar()
# cb.set_label('log(T90)')
'''
plt.scatter(df4['X'], df4['Y'], c='blue', marker='*',
            alpha=1, s=100, label='Fitted GRBs')
plt.scatter(df5['X'], df5['Y'], c='green',
            alpha=0.3, label='GRBs of BAT sample')
plt.xlabel('log(T90)')
plt.legend()
plt.show()

'''


# alt.renderers.enable('altair_viewer')
chart = alt.Chart(df).mark_point(size=80, filled=True, opacity=0.6).encode(
    x='X',
    y='Y',
    color=alt.Color('logT90', scale=alt.Scale(scheme='viridis')),
    tooltip=['GRB_name', 'logT90', 'common_jetfit_tsne']
).interactive()
chart.save('interactive_scatter_color2.html')
# chart.show()
'''
hist_SL = plt.hist([df2['logT90'], df3['logT90']], label=['S', 'L'],
                   color=['blue', 'red'], alpha=0.3, histtype='stepfilled')
plt.legend()
plt.xlabel('log(T90)')
plt.ylabel('counts')
plt.grid(True)
plt.show()
'''
