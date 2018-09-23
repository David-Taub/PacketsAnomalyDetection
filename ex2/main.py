import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import io


# convert txt to csv by removing spaces and hashes, then parse it
def load_data(txt_filepath):
    with open(txt_filepath, 'r') as txt_file:
        raw_data = txt_file.read()
    raw_data = raw_data.replace(' ', '')
    raw_data = raw_data.replace('#', '')
    return pd.read_csv(io.StringIO(raw_data), delimiter=r'|').values


# returns an array of differences between the values of the right 4 bytes
# in packets and their +16 offset packets
def get_rotation_breaks(data):
    payloads = data[:, 3]
    decimal_lsb = np.array([int(payload[-4:], 16) for payload in payloads])
    rotation_diff = decimal_lsb[16:] - decimal_lsb[:-16]
    return rotation_diff


# assumes that a big difference between the least significant 4 bytes in every other 16 packet is rare
# the given attack makes distinguishable differences very often in these bytes
def main():
    dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))
    regular_recording = os.path.join(dirname, r'reference_recording_02.txt')
    regular_recording_img = os.path.join(dirname, r'reference_recording_02.png')
    attacked_recording = os.path.join(dirname, r'attacked_recording_02.txt')
    attacked_recording_img = os.path.join(dirname, r'attacked_recording_02.png')

    data = load_data(regular_recording)
    rotation_diff = get_rotation_breaks(data)
    plt.plot(rotation_diff)
    plt.savefig(regular_recording_img)
    plt.clf()
    data = load_data(attacked_recording)
    rotation_diff = get_rotation_breaks(data)
    plt.plot(rotation_diff)
    plt.savefig(attacked_recording_img)
    plt.clf()


if __name__ == '__main__':
    main()
