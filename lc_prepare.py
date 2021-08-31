import numpy as np
import pandas as pd
import sys
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from decimal import Decimal
from tqdm import tqdm


summary = pd.read_csv(
    '/Users/alessandraberretta/Desktop/summary_burst_durations.txt', delimiter='|')
trig_time = summary[' Trig_time_met '].values
trig_ID = summary[' Trig_ID '].values
GRB_name = summary['## GRBname '].values
T90_start = summary[' T90_start '].values
T90_stop = summary[' T90_stop '].values

len_TIME_list = []

FT_total = []

T90_list = []

path_GRB = '/Users/alessandraberretta/Desktop/lc_64ms_swift_GRB/'
list_file = [path_GRB +
             file for file in os.listdir(path_GRB) if file.endswith('.lc')]

for idx, elm in enumerate(GRB_name):

    T90 = abs(T90_stop[idx] - T90_start[idx])

    T90_list.append(T90)

reference_point = ((np.amax(T90_list))/0.064)*5

for elm in tqdm(list_file): 

    text = elm.split('_')[3]
    trig_id = text[8:-4]
    lc_64 = fits.open(elm)

    data = lc_64[1].data 
        
    # TIME = data['TIME'] - trig_time[idx]
    RATE = data['RATE']
    # ERROR = data['ERROR']

    ene_15_25 = []
    ene_25_50 = []
    ene_50_100 = []
    ene_100_350 = []
    for elm in RATE:
        ene_15_25.append(elm[0])
        ene_25_50.append(elm[1])
        ene_50_100.append(elm[2])
        ene_100_350.append(elm[3])

    list_ene = [ene_15_25, ene_25_50, ene_50_100, ene_100_350]

    padded_channels = []

    for channel in list_ene:
        padded_channels.append(np.pad(
            channel, (0, round(reference_point)-len(channel)), 'constant'))

    total_rate = np.concatenate(
        (padded_channels[0], padded_channels[1], padded_channels[2], padded_channels[3]))

    normalization = np.sum(total_rate)/len(total_rate)

    if normalization: 

        normalized_total_rate = total_rate/normalization

        FT = np.fft.fft(normalized_total_rate)/len(normalized_total_rate)

        FT2 = np.power(np.absolute(FT), 2)

        FT_total.append(FT2)

df = pd.DataFrame(FT_total, columns=[i for i in range(len(FT_total[0]))])

#print(df)

df.to_csv('fft_GRBs.txt', index=False, sep=' ')












'''
lc_070306_64 = (fits.open(
    '/Users/alessandraberretta/Desktop/sw00263361000b_4chan_64ms.lc'), "070306")
lc_070129_64 = (fits.open(
    '/Users/alessandraberretta/Desktop/sw00258408000b_4chan_64ms.lc'), "070129")
lc_070208_64 = (fits.open(
    '/Users/alessandraberretta/Desktop/sw00259714000b_4chan_64ms.lc'), "070208")
lc_070411_64 = (fits.open(
    '/Users/alessandraberretta/Desktop/sw00275087000b_4chan_64ms.lc'), "070411")
lc_070714B_64 = (fits.open(
    '/Users/alessandraberretta/Desktop/sw00284856000b_4chan_64ms.lc'), "070714B")


for lc in [lc_070306_64, lc_070129_64, lc_070208_64, lc_070411_64, lc_070714B_64]:

    data = (lc[0])[1].data

    name = lc[1]

    for idx, elm in enumerate(GRB_name):
        if name in elm:

            T90 = abs(T90_stop[idx] - T90_start[idx])

            T90_list.append(T90)

reference_point = ((np.amax(T90_list))/0.064)*3


for lc in [lc_070306_64, lc_070129_64, lc_070208_64, lc_070411_64, lc_070714B_64]:

    data = (lc[0])[1].data

    name = lc[1]

    for idx, elm in enumerate(GRB_name):
        if name in elm:
            TIME = data['TIME'] - trig_time[idx]
            RATE = data['RATE']
            ERROR = data['ERROR']

            ene_15_25 = []
            ene_25_50 = []
            ene_50_100 = []
            ene_100_350 = []
            for elm in RATE:
                ene_15_25.append(elm[0])
                ene_25_50.append(elm[1])
                ene_50_100.append(elm[2])
                ene_100_350.append(elm[3])

            list_ene = [ene_15_25, ene_25_50, ene_50_100, ene_100_350]
            print(len(ene_15_25))

            padded_channels = []

            for channel in list_ene:
                padded_channels.append(np.pad(
                    channel, (0, round(reference_point)-len(channel)), 'constant'))

            total_rate = np.concatenate(
                (padded_channels[0], padded_channels[1], padded_channels[2], padded_channels[3]))

            normalization = np.sum(total_rate)/len(total_rate)

            normalized_total_rate = total_rate/normalization

            FT = np.fft.fft(total_rate)/len(total_rate)

            FT2 = np.power(np.absolute(FT), 2)

            FT_total.append(FT2)

df = pd.DataFrame({
    'col_1': ['%.6E' % Decimal(x) for x in FT_total[0]], 
    'col_2': ['%.6E' % Decimal(x) for x in FT_total[1]],
    'col_3': ['%.6E' % Decimal(x) for x in FT_total[2]],
    'col_4': ['%.6E' % Decimal(x) for x in FT_total[3]],
    'col_5': ['%.6E' % Decimal(x) for x in FT_total[4]],
})

df_transposed = df.T

# print(df)

df_transposed.to_csv('fft_5_GRBs.txt', index=False)




time = np.arange(0, 100, 0.1)
amplitude = np.sin(time)
FT = (np.fft.fft(amplitude))/len(amplitude)
FT2 = np.power(np.absolute(FT), 2)
# print(FT2)
freq = np.fft.fftfreq(time.shape[-1])
# print(freq)
fig, ax = plt.subplots(2)
ax[0].plot(time, amplitude, color='black')
ax[1].plot(freq, FT2, color='red')
# plt.show()

fig, ax = plt.subplots(4)
ax[0].plot(TIME, ene_15_25, color='black')
ax[1].plot(TIME, ene_25_50, color='red')
ax[2].plot(TIME, ene_50_100, color='green')
ax[3].plot(TIME, ene_100_350, color='blue')
ax[0].set_xlim(-300, 300)
ax[1].set_xlim(-300, 300)
ax[2].set_xlim(-300, 300)
ax[3].set_xlim(-300, 300)
plt.show()
'''
