import os
import sys
import numpy as np
import pandas as pd
import io


def enum_to_onehot(enum):
    print('running enum_to_onehot...')
    values, enum_ids = np.unique(enum, return_inverse=True)
    packet_amount = enum.shape[0]
    bins = len(values)
    onehot_values = np.zeros((packet_amount, bins))
    onehot_values[np.arange(packet_amount), enum_ids] = 1
    return onehot_values


def int_to_onehot(int_column):
    print('running int_to_onehot...')
    packet_amount = int_column.shape[0]
    bins = np.max(int_column)
    ints = int_column.ravel().astype('int64') - 1
    onehot_values = np.zeros((packet_amount, bins))
    onehot_values[np.arange(packet_amount), ints] = 1
    return onehot_values


def hex_to_onehot(data_strings):
    print('running hex_to_onehot...')
    packet_amount = data_strings.shape[0]
    strings_lengths = np.vectorize(len)(data_strings) / 2
    bin_repr_length = int(np.max(strings_lengths)) * 8
    bin_repr = np.vectorize(lambda x: bin(int(x, 16))[2:].zfill(bin_repr_length))(data_strings)
    bin_repr = bin_repr.view('U1').reshape((packet_amount, -1))
    bin_repr = (bin_repr == '1').astype('float')
    return bin_repr


def diff_column(float_column):
    print('running diff_column...')
    MAX_ALLOWED_TIME_DIFF = 1000.0
    SESSION_START_SYMBOL = -1.0
    diff = np.ediff1d(float_column, to_begin=SESSION_START_SYMBOL)
    diff[np.any((diff < 0.0, diff > MAX_ALLOWED_TIME_DIFF), axis=0)] = SESSION_START_SYMBOL
    for i in [7.0, 8.0, 9.0]:
        diff[np.all((diff > i, diff < i + 1))] = i
    diff[diff > 10.0] = 10.0
    diff = np.round((diff * 10).astype('int'))
    return enum_to_onehot(diff)


# transforms the given data into a binary format that should be easier
# for a deep network to train on
def preprocess(data):
    # example row:
    # 000001.100|0200|8|3e9001f43e800dea|2
    # row structure:
    # timestamp, packet type, data_length, data, source
    data = data[:, :-1]
    column_funcs = (diff_column, enum_to_onehot, int_to_onehot, hex_to_onehot, enum_to_onehot)
    column_funcs = (diff_column, enum_to_onehot, int_to_onehot, hex_to_onehot)
    transformed = [column_funcs[i](data[:, [i]]) for i in range(len(column_funcs))]
    processed_data = np.concatenate(transformed, 1)
    print('Feature expanded from %d to %d' % (data.shape[1], processed_data.shape[1]))
    return processed_data.astype('float')


def load_data(txt_filepath):
    with open(txt_filepath, 'r') as txt_file:
        raw_data = txt_file.read()
    raw_data = raw_data.replace(' ', '')
    raw_data = raw_data.replace('#', '')
    return pd.read_csv(io.StringIO(raw_data), delimiter=r'|').values


def main():
    dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))
    regular_recording = os.path.join(dirname, r'reference_recording_01.txt')
    print("Reading TXT...")
    data = load_data(regular_recording)
    print("Preprocessing...")
    processed_data = preprocess(data)
    print("Saving data...")
    np.save(regular_recording.replace('.txt', '.npy'), processed_data)
    print('Done')


if __name__ == '__main__':
    main()

