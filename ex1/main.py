import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import io


# convert txt to csv by removing spaces and hashes, then parse it
def load_data(txt_filepath):
    with open(txt_filepath, 'r') as txt_file:
        raw_data = txt_file.read()
    raw_data = raw_data.replace(' ', '')
    raw_data = raw_data.replace('#', '')
    return pd.read_csv(io.StringIO(raw_data), delimiter=r'|').values


# returns diff between values in column
# replaces values too high and too low with a negative value that marks a new session
def diff_column(float_column):
    MAX_ALLOWED_TIME_DIFF = 1000.0
    SESSION_START_SYMBOL = -1.0
    diff = np.ediff1d(float_column, to_begin=SESSION_START_SYMBOL)
    diff[np.any((diff < 0.0, diff > MAX_ALLOWED_TIME_DIFF), axis=0)] = SESSION_START_SYMBOL
    return diff


# splits recording to streams by the type column
# bandwidth of payload is calculated per packet type
# generates figure and saves as image file
def gen_streamed_bandwidth_figure(txt_filepath, img_out_filepath):
    EPS = 0.001
    data = load_data(txt_filepath)
    packet_types = np.unique(data[:, 1])
    for packet_type in packet_types:
        stream_mask = data[:, 1] == packet_type
        time_diff = diff_column(data[stream_mask, 0])
        data_lengths = data[stream_mask, 2]
        # EPS added to avoid zero division
        bandwidth = data_lengths / (time_diff + EPS)
        # remove negatives, which are the end \ start session marks
        bandwidth = bandwidth[bandwidth > 0]
        plt.plot(bandwidth)
    plt.ylabel('bytes/millisec')
    plt.xlabel('Packet in Stream')
    plt.legend(packet_types)
    plt.savefig(img_out_filepath)
    plt.clf()


def main():
    dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))
    regular_recording = os.path.join(dirname, r'reference_recording_01.txt')
    regular_recording_img = os.path.join(dirname, r'reference_recording_01.png')
    attacked_recording = os.path.join(dirname, r'attacked_recording_01.txt')
    attacked_recording_img = os.path.join(dirname, r'attacked_recording_01.png')

    gen_streamed_bandwidth_figure(regular_recording, regular_recording_img)
    gen_streamed_bandwidth_figure(attacked_recording, attacked_recording_img)


if __name__ == '__main__':
    main()
