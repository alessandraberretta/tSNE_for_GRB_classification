import numpy as np
import pandas as pd
import sys
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from decimal import Decimal
from tqdm import tqdm
from sklearn import manifold, datasets
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import pandas as pd
import pylab
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

path_GRB = '/Users/alessandraberretta/tSNE_for_GRB_classification/lc_64ms_swift_GRB/'

list_file = [path_GRB +
             file for file in os.listdir(path_GRB) if file.endswith('.lc')]

for idx, elm in enumerate(GRB_name):

    T90 = abs(T90_stop[idx] - T90_start[idx])

    T90_list.append(T90)

reference_point = ((np.amax(T90_list))/0.064)*3


def gen_data():

    for elm in tqdm(list_file, desc="gen data"):

        # trig_id = elm.split('/')[3][4:-18]
        trig_id = elm.split('_')[6][8:-4]

        for idx, trig in enumerate(trig_ID):
            if trig_id == str(trig):
                T90_single = abs(T90_stop[idx] - T90_start[idx])

        T90_GRBs.append(T90_single)

        # text = elm.split('_')[3]
        # trig_id = text[8:-4]

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

            # FT_total.append(FT2)
            yield pd.Series(FT2, index=[f"col{col:02d}" for col in list(range(len(FT2)))])


df = pd.DataFrame(gen_data())

# print(df)

# df.to_csv('fft_GRBs.txt', index=False, sep=' ')
# df.to_parquet("fft_GRBs.parquet")

# X = np.loadtxt("/Users/alessandraberretta/Desktop/fft_GRBs.txt")

tsne = manifold.TSNE()
X_embedded = tsne.fit_transform(df)

df2 = pd.DataFrame(
    {"x": X_embedded[:, 0], "y": X_embedded[:, 0], "z": np.log10(T90_GRBs)})

brush = alt.selection(type="interval")
points = (
    alt.Chart(df2)
    .mark_point()
    .encode(x="x:Q", y="y:Q", color='z')).interactive()

points.show()

# scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 0])
# plt.colorbar(scatter)
# plt.show()


'''
def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i],
                                      (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y


if __name__ == "__main__":
    # print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    # print("Running example on 2,500 MNIST digits...")
    # X = np.loadtxt("/Users/alessandraberretta/Desktop/fft_5_GRBs.txt")
    # labels = np.loadtxt("mnist2500_labels.txt")
    Y = tsne(df, 2, 50, 20.0)
    pylab.scatter(Y[:, 0], Y[:, 1], 20)
    # pylab.savefig()
    pylab.show()



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
