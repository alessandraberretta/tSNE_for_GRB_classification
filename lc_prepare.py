import numpy as np
import pandas as pd
import sys
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from decimal import Decimal
from tqdm import tqdm
from sklearn import manifold, datasets
import altair as alt


summary = pd.read_csv(
    '/Users/alessandraberretta/tSNE_for_GRB_classification/summary_burst_durations.txt', delimiter='|')
trig_time = summary[' Trig_time_met '].values
trig_ID = summary[' Trig_ID '].values
GRB_name = summary['## GRBname '].values
T90_start = summary[' T90_start '].values
T90_stop = summary[' T90_stop '].values

len_TIME_list = []

FT_total = []

T90_list = []

T90_GRBs = []

temporal_class = []

path_GRB = '/Users/alessandraberretta/tSNE_for_GRB_classification/lc_64ms_swift_GRB/'

list_file = [path_GRB +
             file for file in os.listdir(path_GRB) if file.endswith('.lc')]

TF_before = False

FT_rate = False

single_channel = False

standard = True

FT2_channels = []

normalized_total_rate_GRBs = []


for idx, elm in enumerate(GRB_name):

    T90 = abs(T90_stop[idx] - T90_start[idx])

    T90_list.append(T90)

reference_point = np.amax(T90_list)/0.064


def nextpower(num, base):
    i = 1
    while i < num:
        i *= base
    return i


pow2 = nextpower(reference_point, 2)
# print(pow2)

RATE_GRBs = []

'''
for elm in tqdm(list_file):

    lc_64 = fits.open(elm)

    data = lc_64[1].data

    RATE = data['RATE']

    RATE_GRBs.append(len(RATE))

ref_point = np.amax(RATE_GRBs) + 100
'''


def gen_data():

    for elm in tqdm(list_file, desc="gen data"):

        trig_id = elm.split('_')[6][8:-4]

        t90start = T90_start[np.where(trig_ID == float(trig_id))]
        t90stop = T90_stop[np.where(trig_ID == float(trig_id))]
        trigtime = trig_time[np.where(trig_ID == float(trig_id))]

        lc_64 = fits.open(elm)

        data = lc_64[1].data

        TIME = data['TIME']
        RATE = data['RATE']

        # df_lc = pd.DataFrame(TIME, RATE, columns=['Time', 'Rate'])

        # print(df_lc)

        # ERROR = data['ERROR']

        ene_15_25 = []
        ene_25_50 = []
        ene_50_100 = []
        ene_100_350 = []

        for idx, elm in enumerate(RATE):

            if (t90start+trigtime)-0.1*abs(t90stop - t90start) < TIME[idx] < (t90stop+trigtime)+0.1*abs(t90stop - t90start):

                if not np.all(elm == 0):
                    ene_15_25.append(elm[0])
                    ene_25_50.append(elm[1])
                    ene_50_100.append(elm[2])
                    ene_100_350.append(elm[3])

        # dict_lc = {'Time': TIME, 'Rate_15_25': ene_15_25, 'Rate_25_50': ene_25_50,
        # 'Rate_50_100': ene_50_100, 'Rate_100_350': ene_100_350}

        # df_lc = pd.DataFrame(dict_lc)

        list_ene = [ene_15_25, ene_25_50, ene_50_100, ene_100_350]

        padded_channels = []

        for channel in list_ene:
            padded_channels.append(np.pad(
                channel, (0, pow2-len(channel)), 'constant'))

        if single_channel:

            normalization = np.sum(padded_channels[0])

            if normalization:

                for idx, trig in enumerate(trig_ID):

                    if trig_id == str(trig):

                        T90_single = abs(T90_stop[idx] - T90_start[idx])

                T90_GRBs.append(T90_single)

                normalized_chan = padded_channels[0]/normalization

                # FT = np.fft.fft(normalized_chan)/len(normalized_chan)

                # FT2 = np.power(np.absolute(FT), 2)

                yield pd.Series(normalized_chan, index=[f"col{col:02d}" for col in list(range(len(normalized_chan)))])

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

            normalization = np.sum(total_rate)
            # /len(total_rate)

            if normalization:

                for idx, trig in enumerate(trig_ID):

                    if trig_id == str(trig):

                        T90_single = abs(T90_stop[idx] - T90_start[idx])

                T90_GRBs.append(T90_single)

                normalized_total_rate = total_rate/normalization

                if FT_rate:

                    # normalized_total_rate_GRBs.append(normalized_total_rate)

                    yield pd.Series(normalized_total_rate, index=[f"col{col:02d}" for col in list(range(len(normalized_total_rate)))])

                else:

                    FT = np.fft.fft(normalized_total_rate) / \
                        len(normalized_total_rate)

                    FT2 = np.power(np.absolute(FT), 2)

                    # FT_total.append(FT2)
                    yield pd.Series(FT2, index=[f"col{col:02d}" for col in list(range(len(FT2)))])


df = pd.DataFrame(gen_data())
# df2 = pd.DataFrame(T90_GRBs)
# df3 = df.append(df2, sort=False)
# print(temporal_class)


# df.to_csv('fft_GRBs.txt', index=False, sep=' ')
# df.to_parquet("fft_GRBs.parquet")

# X = np.loadtxt("/Users/alessandraberretta/Desktop/fft_GRBs.txt")

tsne = manifold.TSNE(perplexity=30.0)
X_embedded = tsne.fit_transform(df)


fig, ax = plt.subplots()
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=(
    np.log10(T90_GRBs)), cmap='viridis')
# ax.set_xlim(-40, 40)
# ax.set_ylim(-40, 40)
plt.colorbar()
plt.show()
