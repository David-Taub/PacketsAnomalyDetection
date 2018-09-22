from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.cluster


txt_filepath = r'C:\Users\booga\Dropbox\projects\argus\Argus_Protocol_Research\reference_recording_01.txt'
ENCODER_MODEL_FILEPATH = r'C:\Users\booga\Dropbox\projects\argus\Argus_Protocol_Research\encoder.h5'
X_TRAIN_MM_FILEPATH = r'C:\Users\booga\Dropbox\projects\argus\Argus_Protocol_Research\x_train.mm'
Y_TRAIN_MM_FILEPATH = r'C:\Users\booga\Dropbox\projects\argus\Argus_Protocol_Research\y_train.mm'
X_VALIDATION_MM_FILEPATH = r'C:\Users\booga\Dropbox\projects\argus\Argus_Protocol_Research\x_validation.mm'
Y_VALIDATION_MM_FILEPATH = r'C:\Users\booga\Dropbox\projects\argus\Argus_Protocol_Research\y_validation.mm'
PACKET_FEATURES = 80
SLIDING_WINDOW_WIDTH = 20



def convert_txt_to_csv(txt_filepath, csv_filepath):
    with open(txt_filepath, 'r') as txt_file:
        raw_data = txt_file.read()
    raw_data = raw_data.replace(' ', '')
    raw_data = raw_data.replace('#', '')
    with open(csv_filepath, 'w') as csv_file:
        csv_file.write(raw_data)


def enum_to_onehot(enum):
    values, enum_ids = np.unique(enum, return_inverse=True)
    packet_amount = enum.shape[0]
    bins = len(values)
    onehot_values = np.zeros((packet_amount, bins))
    onehot_values[enum_ids] = 1
    return onehot_values


def int_to_onehot(int_column):
    packet_amount = int_column.shape[0]
    bins = np.max(int_column)
    onehot_values = np.zeros((packet_amount, bins))
    onehot_values[int_column.astype('int')] = 1.0
    return onehot_values


def hex_to_onehot(data_strings):
    packet_amount = data_strings.shape[0]
    data_strings = data[:, 3]
    strings_lengths = np.vectorize(len)(data_strings) / 2
    bin_repr_length = int(np.max(strings_lengths)) * 8
    bin_repr = np.vectorize(lambda x: bin(int(x, 16))[2:].zfill(bin_repr_length))(data_strings)
    bin_repr = bin_repr.view('U1').reshape((packet_amount, -1))
    bin_repr = (bin_repr == '1').astype('float')
    return bin_repr


def diff_column(float_column):
    MAX_ALLOWED_TIME_DIFF = 100.0
    SESSION_START_SYMBOL = -1.0
    diff = np.ediff1d(float_column, to_begin=SESSION_START_SYMBOL)
    diff[np.any((diff < 0.0, diff > MAX_ALLOWED_TIME_DIFF), axis=0)] = SESSION_START_SYMBOL
    diff = np.expand_dims(diff, 1)
    return diff


def preprocess(data):
    # example row:
    # 000001.100|0200|8|3e9001f43e800dea|2
    # row structure:
    # timestamp, packet type, data_length, data, source
    column_funcs = (diff_column, enum_to_onehot, int_to_onehot, hex_to_onehot, enum_to_onehot)
    transformed = [column_funcs[i](data[:, [i]]) for i in range(data.shape[1])]
    processed_data = np.concatenate(transformed, 1)
    return processed_data.astype('float')


def load_data(txt_filepath):
    csv_filepath = txt_filepath.replace('.txt', '.csv')
    convert_txt_to_csv(txt_filepath, csv_filepath)
    data = pd.read_csv(csv_filepath, delimiter=r'|').values
    return data

def gen_sequences(processed_data):
    N = processed_data.shape[0]
    x_train = processed_data[:int(N*0.9), 1:]
    x_validation = processed_data[int(N*0.9):, 1:]
    x_train_seqs = np.memmap(X_TRAIN_MM_FILEPATH, dtype='float32', mode='w', shape=(N - SLIDING_WINDOW_WIDTH, SLIDING_WINDOW_WIDTH-1, PACKET_FEATURES))
    y_train_seqs = np.memmap(Y_TRAIN_MM_FILEPATH, dtype='float32', mode='w', shape=(N - SLIDING_WINDOW_WIDTH, PACKET_FEATURES))
    x_validation_seqs = np.memmap(X_VALIDATION_MM_FILEPATH, dtype='float32', mode='w', shape=(N - SLIDING_WINDOW_WIDTH, SLIDING_WINDOW_WIDTH-1, PACKET_FEATURES))
    y_validation_seqs = np.memmap(Y_VALIDATION_MM_FILEPATH, dtype='float32', mode='w', shape=(N - SLIDING_WINDOW_WIDTH, PACKET_FEATURES))

    for i in tqdm.tqdm(range(x_train.shape[0] - SLIDING_WINDOW_WIDTH)):
        x_train_seqs[i, :, :] = x_train[i:i + SLIDING_WINDOW_WIDTH - 1, :]
        y_train_seqs[i, :] = x_train[i + SLIDING_WINDOW_WIDTH - 1, :]


    for i in tqdm.tqdm(range(x_validation.shape[0] - SLIDING_WINDOW_WIDTH)):
        x_validation_seqs[i, :, :] = x_validation[i:i + SLIDING_WINDOW_WIDTH - 1, :]
        y_validation_seqs[i, :] = x_validation[i + SLIDING_WINDOW_WIDTH - 1, :]

    # close file handles
    x_validation_seqs._mmap.close()
    y_validation_seqs._mmap.close()
    x_train_seqs._mmap.close()
    y_train_seqs._mmap.close()


def main():
    print("Reading CSV...")
    data = load_data(txt_filepath)
    print("Preprocessing...")
    processed_data = preprocess(data)
    print("Saving data...")
    np.save(txt_filepath.replace('.txt', '.npy'), processed_data)
    print("Generating sequences...")
    gen_sequences(processed_data)
    print('Done')

if __name__ == '__main__':
    main()
