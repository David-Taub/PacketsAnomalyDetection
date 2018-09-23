import tqdm
import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.cluster
import copy
import matplotlib.pyplot as plt

TXT_FILEPATH = r'C:\Users\booga\Dropbox\projects\PacketsAnomalyDetection\Argus_Protocol_Research\reference_recording_01.txt'
ENCODING_DIM = 60
SLIDING_WINDOW_WIDTH = 50


# processed_data = np.load(TXT_FILEPATH.replace('.txt', '2.npy'))
for t in range(7):
    data = np.load(TXT_FILEPATH.replace('.txt', 'raw_%d.npy' % t))
    processed_data = np.load(TXT_FILEPATH.replace('.txt', '%d.npy' % t))
    time_diff = data[:, 2] / (processed_data[:, 0] + 0.01)
    cumsum = time_diff.copy()
    for i in range(1, SLIDING_WINDOW_WIDTH):
        shifted = np.roll(time_diff, i)
        shifted[:i] = 0.0
        cumsum += shifted
    plt.plot(cumsum)

for t in range(7):
    data = np.load(TXT_FILEPATH.replace('.txt', 'raw_%d_attack.npy' % t))
    processed_data = np.load(TXT_FILEPATH.replace('.txt', '%d_attack.npy' % t))
    time_diff = data[:, 2] / (processed_data[:, 0] + 0.01)
    cumsum = time_diff.copy()
    for i in range(1, SLIDING_WINDOW_WIDTH):
        shifted = np.roll(time_diff, i)
        shifted[:i] = 0.0
        cumsum += shifted
    plt.plot(cumsum)

plt.ylabel('time to broadcast %d packets' % SLIDING_WINDOW_WIDTH)
plt.show()