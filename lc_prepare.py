from turtle import color
import numpy as np
import pandas as pd
import sys
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from decimal import Decimal
from tqdm import tqdm
from sklearn import manifold, datasets
import scipy
import plotly.express as px

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
collapsar_merger = []
GRB_review = []

TF_before = False
FT_rate = False
single_channel = True
standard = False
long_subclass = False

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
            # print('ok ')
        elif trig_id == '147478':
            t90start = 0
            t90stop = 3
            # print('ok ')
        elif trig_id == '284856':
            t90start = 0
            t90stop = 2.0
            # print('ok ')
        elif trig_id == '232585':
            t90start = 0
            t90stop = 0.4
            # print('ok ')
        elif trig_id == '243690':
            t90start = 0
            t90stop = 0.2
            # print('ok ')
        elif trig_id == '299787':
            t90start = 0
            t90stop = 1.8
            # print('ok ')
        elif trig_id == '351588':
            t90start = 0
            t90stop = 0.3
            # print('ok ')

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
            # print('all 0s datafile:', GRBname[0])
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
                # print('GRB080503    ')
            elif GRBname[0] == 'GRB050724    ':
                T90_single = 3.0
                # print('GRB050724    ')
            elif GRBname[0] == 'GRB070714B   ':
                T90_single = 2.0
                # print('GRB070714B   ')
            elif GRBname[0] == 'GRB061006   ':
                T90_single = 0.4
                # print('GRB061006   ')
            elif GRBname[0] == 'GRB061210   ':
                T90_single = 0.2
                # print('GRB061210   ')
            elif GRBname[0] == 'GRB071227   ':
                T90_single = 1.8
                # print('GRB071227   ')
            elif GRBname[0] == 'GRB090510   ':
                T90_single = 0.3
                # print('GRB090510   ')
            else:
                for idx, trig in enumerate(trig_ID):

                    if trig_id == str(trig):

                        T90_single = abs(T90_stop[idx] - T90_start[idx])

            # if GRBname[0].startswith('GRB16') or GRBname[0].startswith('GRB17') or GRBname[0].startswith('GRB18')\
                    # or GRBname[0].startswith('GRB19') or GRBname[0].startswith('GRB20'):
                # continue

            T90_GRBs.append(T90_single)
            names.append(GRBname[0])

            if GRBname[0].startswith('GRB2112'):
                GRB_review.append('no')
            else:
                GRB_review.append('no')

            '''
            if T90_single < 0.8:
                collapsar_merger.append('merger')
            else:
                collapsar_merger.append('collapsar')
            '''

            FT = np.fft.fft(padded_channels[0])/len(padded_channels[0])

            FT2 = np.power(np.absolute(FT), 2)
            
            if GRBname[0].startswith('GRB070809'):
                print('found GRB070809')
                plt.plot(FT2, color='maroon')
                plt.xlim(1,len(FT2/2))
                plt.xscale("log")
                # plt.title("GRB070809")
                # plt.savefig('power_spectrum_GRB070809.png')
                # plt.close()
            if GRBname[0].startswith('GRB050724'):
                print('found GRB050724')
                plt.plot(FT2, color='maroon')
                plt.xlim(1,len(FT2/2))
                plt.xscale("log")
                # plt.title("GRB050724")
                # plt.savefig('power_spectrum_GRB050724.png')
                # plt.close()
            if GRBname[0].startswith('GRB070714B'):
                print('found GRB070714B')
                plt.plot(FT2, color='maroon')
                plt.xlim(1,len(FT2/2))
                plt.xscale("log")
                # plt.title("GRB070714B")
                # plt.savefig('power_spectrum_GRB070714B.png')
                # plt.close()
            if GRBname[0].startswith('GRB130603B'):
                print('found GRB130603B')
                plt.plot(FT2, color='red')
                plt.xlim(1,len(FT2/2))
                plt.xscale("log")
                # plt.title("GRB130603B")
                # plt.savefig('power_spectrum_GRB130603B.png')
                # plt.close()
            if GRBname[0].startswith('GRB080503'):
                print('found GRB080503')
                plt.plot(FT2, color='red')
                plt.xlim(1,len(FT2/2))
                plt.xscale("log")
                # plt.title("GRB080503")
                # plt.savefig('power_spectrum_GRB080503.png')
                # plt.close()
            if GRBname[0].startswith('GRB160821B'):
                print('found GRB160821B')
                plt.plot(FT2, color='red')
                plt.xlim(1,len(FT2/2))
                plt.xscale("log")
                # plt.title("GRB160821B")
                # plt.savefig('power_spectrum_GRB160821B.png')
                # plt.close()
            if GRBname[0].startswith('GRB211211A'):
                print('found GRB160821B')
                plt.plot(FT2, color='maroon')
                plt.xlim(1,len(FT2/2))
                plt.xscale("log")
                plt.xlabel(r'|DTFT|$^{2}$')
                plt.ylabel('counts')
                plt.title("Power spectrum of kilonova-GRBs")
                # plt.savefig('power_spectrum_GRB160821B.png')
                # plt.close()
            
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

            # total_rate = np.concatenate(
                # (padded_channels[0], padded_channels[1]))

            total_rate = np.concatenate(
                (padded_channels[0], padded_channels[1], padded_channels[2], padded_channels[3]))

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

            # if GRBname[0].startswith('GRB16') or GRBname[0].startswith('GRB17') or GRBname[0].startswith('GRB18')\
                    # or GRBname[0].startswith('GRB19') or GRBname[0].startswith('GRB20'):
                # continue

            if GRBname[0].startswith('GRB2112'):
                GRB_review.append('yes')
                print('CIAO')
            else:
                GRB_review.append('no')

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

contamination_Bromberg = ['050709', '050925', '060313', '060502B', '060801', '061201', '061217', '070209', 
                        '070923', '071112B', '080919', '090305A', '090417A', '090510', '090515', '090621B', 
                        '090815C', '091109B', '100117A', '100206A', '100625A', '100628A', '101129A', '101219A', 
                        '110420B', '111117A', '111126A']

grb_KN = ['050709A', '050724', '070714B',
          '070809A', '080503', '130603B', '150101B', '160821B', '070809', '211211A']
# '060614', 
KN = []
notcollapsar = []

for kn in names_for_comparison:
    if kn in grb_KN:
        KN.append('kn')
    else:
        KN.append('nokn')

for burst in names_for_comparison:
    if burst in contamination_Bromberg:
        notcollapsar.append('yes')
    else:
        notcollapsar.append('no')


df_check = pd.DataFrame()
df_check['name'] = names_for_comparison
df_check.to_csv('check.csv', index=False, sep='\t')

df_comparison = pd.read_csv('/Users/alessandraberretta/JetFit/common_grb.csv')
common = df_comparison['common_grb'].values

# tsne = manifold.TSNE(perplexity=30.0, random_state=1, init='pca')
tsne = manifold.TSNE(perplexity=30.0, random_state=0, init='pca')
X_embedded = tsne.fit_transform(df)

my_class = []
X_embedded_x = []
X_embedded_y = []

for idx, elm in enumerate(X_embedded):

    X_embedded_x.append(elm[0])
    # X_embedded_y.append(-elm[1])
    X_embedded_y.append(-elm[1])

    # if elm[0] < 2.5 and elm[0] > -16 and elm[1] < -15 and elm[1] > -45:
    if elm[0] > -38 and elm[0] < -14 and -elm[1] > -35 and -elm[1] < -12:
        my_class.append('type_S')
    else:
        my_class.append('type_L')

df['X'] = X_embedded_x
df['Y'] = X_embedded_y
df['classification'] = my_class
df['logT90'] = np.log10(T90_GRBs)
df['T90'] = T90_GRBs
df['GRB_name'] = names_for_comparison
df['KN'] = KN
df['notcollapsar'] = notcollapsar
df['GRB_review'] = GRB_review

for idx_, elm_ in enumerate(X_embedded_x):

    # if elm[0] < 2.5 and elm[0] > -16 and elm[1] < -15 and elm[1] > -45:
    if elm_ >= 32.0 and elm_ <= 40.0 and X_embedded_y[idx_] > -16 and X_embedded_y[idx_] < -8:
        print(names_for_comparison[idx_])


'''
df_multiplot = pd.DataFrame()
df_multiplot['X'] = X_embedded_x
df_multiplot['Y'] = X_embedded_y
df_multiplot['log(T90)'] = np.log10(T90_GRBs)
df_multiplot['KN'] = KN
df_multiplot['notcollapsar'] = notcollapsar
df_multiplot.to_csv('multiplot.csv', index=False, sep='\t')
'''

for grb in names_for_comparison:
    if grb.startswith('16') or grb.startswith('17') or grb.startswith('18') or grb.startswith('19') or grb.startswith('20'):
        # print(grb)
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

'''
df_long = pd.DataFrame()
df_long['name'] = df3['GRB_name']
df_long['X'] = df3['X']
df_long['Y'] = df3['Y']
df_long['log(T90)'] = df3['logT90']
df_long.to_csv('long_sub.csv', index=False, sep='\t')
print(len(df_long))

long_sub_file = pd.read_csv(
    '/Users/alessandraberretta/tSNE_for_GRB_classification/long_sub.csv', delimiter='\t')
namelong = long_sub_file['name']
dfsummaryreduced = pd.DataFrame()
for elmname in namelong:
    for elmname2 in GRB_name:
        if elmname in elmname2:
            new_row = summary[summary['## GRBname '].str.contains(elmname)]
            dfsummaryreduced = dfsummaryreduced.append(new_row)
dfsummaryreduced.to_csv('summary_reduced.csv', index=False)
'''

df2 = df[df['classification'] == 'type_S']
df3 = df[df['classification'] == 'type_L']
df4 = df[df['common_jetfit_tsne'] == 'yes']
df5 = df[df['common_jetfit_tsne'] == 'no']
df6 = df4[df4['classification'] == 'type_S']
df7 = df[df['KN'] == 'kn']
df8 = df[df['new_grb'] == 'yes']
df9 = df[df['notcollapsar'] == 'yes']
df10 = df[df['T90'] < 2]
df11 = df2[df2['T90'] < 2]
df12 = df3[df3['T90'] < 2]
df13 = df[df['T90'] > 2]
df14 = df2[df2['T90'] > 2]
df15 = df3[df3['T90'] > 2]
df16 = df[df['GRB_review'] == 'yes']
df17 = df[df['GRB_name'] == '090510']

fig, ax = plt.subplots()
# ax = plt.gca()
# plt.scatter(X_embedded_x, X_embedded_y, c=(
    # np.log10(T90_GRBs)), cmap='viridis', label='GRBs of BAT sample')
# ax = fig.add_subplot(projection='3d')
'''
hist, xedges, yedges, _ = plt.hist2d(X_embedded_x, X_embedded_y, bins=(5, 5), range=[[-40, 40], [-40, 40]], cmap='plasma')
ax.axes.yaxis.set_ticklabels([])
ax.axes.xaxis.set_ticklabels([])
cb = plt.colorbar()
cb.set_label('counts')
'''

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
hist, xedges, yedges = np.histogram2d(X_embedded_x, X_embedded_y, bins=5, range=[[-40, 40], [-40, 40]])
x_ind, y_ind = np.unravel_index(np.argmax(hist), hist.shape)
print(f'The maximum count is {hist[x_ind][y_ind]:.0f} at index ({x_ind}, {y_ind})')
print(f'Between x values {xedges[x_ind]} and {xedges[x_ind+1]}')
print(f'and between y values {yedges[y_ind]} and {yedges[y_ind+1]}')


xpos, ypos = np.meshgrid(xedges[:-1] + 0.01, yedges[:-1] + 0.01, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the 16 bars.
dx = dy = 6 * np.ones_like(zpos)
dz = hist.ravel()

cmap = cm.get_cmap('plasma')
norm = Normalize(vmin=min(dz), vmax=max(dz))
colors = cmap(norm(dz))

sc = cm.ScalarMappable(cmap=cmap,norm=norm)
sc.set_array([])
plt.colorbar(sc, label='counts')

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors)

# fig = px.density_heatmap(df, x="X", y="Y", nbinsx=70, nbinsy=70, marginal_x="histogram", marginal_y="histogram")
# fig.show()

'''
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

dx = dy = 1 * np.ones_like(zpos)
dz = hist.ravel()

cmap = cm.get_cmap('plasma')
norm = Normalize(vmin=min(dz), vmax=max(dz))
colors = cmap(norm(dz))

sc = cm.ScalarMappable(cmap=cmap,norm=norm)
sc.set_array([])
plt.colorbar(sc)

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors)
'''
plt.show()
# plt.scatter(df9['X'], df9['Y'], c='magenta', marker='*', edgecolors='black', alpha=0.99, s=1000, label='non-collapsar GRBs')
# plt.scatter(df16['X'], df16['Y'], c='black', marker='*', alpha=0.99, s=400)
# plt.scatter(df4['X'], df4['Y'], c='blue', marker='*',
            # alpha=0.8, s=300, label='Fitted GRBs')
# plt.scatter(df7['X'], df7['Y'], c='red', marker='^',
            # alpha=0.8, s=300, label='GRBs with KiloNova')
# plt.scatter(df17['X'], df17['Y'], c='red', marker='^',
            # alpha=0.8, sizes=300, label='GRBs with KiloNova')
# plt.scatter(df4['X'], df4['Y'], c='blue', marker='*',
# alpha=0.7, s=80, label='Fitted GRBs')
# plt.scatter(df8['X'], df8['Y'], c='purple', marker='*',
# alpha=0.8, label='new GRBs')
# plt.legend()
# cb = plt.colorbar()
# cb.set_label('log(T90)')
# plt.arrow(0, 3, 10, 10, color='red')
# ax.axes.yaxis.set_ticklabels([])
# plt.show()
'''
plt.scatter(df4['X'], df4['Y'], c='blue', marker='*',
            alpha=0.8, s=80, label='Fitted GRBs')
plt.scatter(df5['X'], df5['Y'], c='green',
            alpha=0.3, label='GRBs of BAT sample')
plt.scatter(df7['X'], df7['Y'], c='red', marker='^',
            alpha=0.8, label='GRBs with KiloNova')
# print(len(df8['X']))
# plt.scatter(df8['X'], df8['Y'], c='purple', marker='^',
# alpha=0.8, label='GRBs 2016-2020')
# plt.xlabel('log(T90)')
plt.legend()
plt.show()
# alt.renderers.enable('altair_viewer')
chart = alt.Chart(df).mark_point(size=80, filled=True, opacity=0.6).encode(
    x='X',
    y='Y',
    color=alt.Color('logT90', scale=alt.Scale(scheme='viridis')),
    tooltip=['GRB_name', 'logT90', 'common_jetfit_tsne', 'KN']
)
chart.save('interactive_scatter_color4.html')
# save(chart, "interactive_scatter_color3.png")
# chart.show()

ciaone = []
for id, elm in enumerate(df3['logT90'].values):
    print(elm, df['GRB_name'][id])

hist_SL = plt.hist([df2['logT90'], df3['logT90']], label=['small group', 'big group'],
                   color=['purple', 'goldenrod'], alpha=0.4, histtype ='stepfilled')

plt.legend()
plt.xlabel('log(T90)')
plt.ylabel('counts')
plt.grid(True)
plt.show()
'''