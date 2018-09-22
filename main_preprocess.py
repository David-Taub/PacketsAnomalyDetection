import tqdm
import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.cluster


TXT_FILEPATH = r'C:\Users\booga\Dropbox\projects\argus\Argus_Protocol_Research\reference_recording_01.txt'
ENCODER_MODEL_FILEPATH = r'C:\Users\booga\Dropbox\projects\argus\Argus_Protocol_Research\encoder.h5'
X_TRAIN_MM_FILEPATH = r'C:\Users\booga\Dropbox\projects\argus\Argus_Protocol_Research\x_train.mm'
Y_TRAIN_MM_FILEPATH = r'C:\Users\booga\Dropbox\projects\argus\Argus_Protocol_Research\y_train.mm'
X_VALIDATION_MM_FILEPATH = r'C:\Users\booga\Dropbox\projects\argus\Argus_Protocol_Research\x_validation.mm'
Y_VALIDATION_MM_FILEPATH = r'C:\Users\booga\Dropbox\projects\argus\Argus_Protocol_Research\y_validation.mm'
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
    strings_lengths = np.vectorize(len)(data_strings) / 2
    bin_repr_length = int(np.max(strings_lengths)) * 8
    bin_repr = np.vectorize(lambda x: bin(int(x, 16))[2:].zfill(bin_repr_length))(data_strings)
    bin_repr = bin_repr.view('U1').reshape((packet_amount, -1))
    bin_repr = (bin_repr == '1').astype('float')
    return bin_repr


def diff_column(float_column):
    MAX_ALLOWED_TIME_DIFF = 100.0
    SESSION_START_SYMBOL = 0.0
    diff = np.ediff1d(float_column, to_begin=SESSION_START_SYMBOL)
    diff[np.any((diff < 0.0, diff > MAX_ALLOWED_TIME_DIFF), axis=0)] = SESSION_START_SYMBOL
    for i in [7.0, 8.0, 9.0]:
        diff[np.all((diff > i, diff < i + 1))] = i
    diff[diff > 10.0] = 10.0
    diff = np.round((diff * 10).astype('int'))
    return enum_to_onehot(diff)


def preprocess(data):
    # example row:
    # 000001.100|0200|8|3e9001f43e800dea|2
    # row structure:
    # timestamp, packet type, data_length, data, source
    column_funcs = (diff_column, enum_to_onehot, int_to_onehot, hex_to_onehot, enum_to_onehot)
    transformed = [column_funcs[i](data[:, [i]]) for i in range(data.shape[1])]
    processed_data = np.concatenate(transformed, 1)
    print('Feature expanded from %d to %d' % (data.shape[1], processed_data.shape[1]))
    return processed_data.astype('float')


def load_data(txt_filepath):
    csv_filepath = txt_filepath.replace('.txt', '.csv')
    convert_txt_to_csv(txt_filepath, csv_filepath)
    data = pd.read_csv(csv_filepath, delimiter=r'|').values
    return data

def gen_sequences(processed_data):
    packet_features = processed_data.shape[1]
    validation_ratio = 0.1
    train_size = int((processed_data.shape[0] - SLIDING_WINDOW_WIDTH) * (1 - validation_ratio))
    validation_size = (processed_data.shape[0] - SLIDING_WINDOW_WIDTH) - train_size
    x_train_seqs = np.memmap(X_TRAIN_MM_FILEPATH, dtype='float32', mode='write', shape=(train_size, SLIDING_WINDOW_WIDTH - 1, packet_features))
    y_train_seqs = np.memmap(Y_TRAIN_MM_FILEPATH, dtype='float32', mode='write', shape=(train_size, packet_features))
    x_validation_seqs = np.memmap(X_VALIDATION_MM_FILEPATH, dtype='float32', mode='write', shape=(validation_size, SLIDING_WINDOW_WIDTH - 1, packet_features))
    y_validation_seqs = np.memmap(Y_VALIDATION_MM_FILEPATH, dtype='float32', mode='write', shape=(validation_size, packet_features))

    order = np.arange(processed_data.shape[0] - SLIDING_WINDOW_WIDTH)[:]
    np.random.shuffle(order)
    for i in tqdm.tqdm(range(train_size)):
        j = order[i]
        x_train_seqs[i, :, :] = processed_data[j:j + SLIDING_WINDOW_WIDTH - 1, :]
        y_train_seqs[i, :] = processed_data[j + SLIDING_WINDOW_WIDTH - 1, :]

    for i in tqdm.tqdm(range(validation_size)):
        j = order[train_size + i]
        x_validation_seqs[i, :, :] = processed_data[j:j + SLIDING_WINDOW_WIDTH - 1, :]
        y_validation_seqs[i, :] = processed_data[j + SLIDING_WINDOW_WIDTH - 1, :]

    # close file handles
    x_validation_seqs._mmap.close()
    y_validation_seqs._mmap.close()
    x_train_seqs._mmap.close()
    y_train_seqs._mmap.close()



def main():
    print("Reading CSV...")
    data = load_data(TXT_FILEPATH)
    print("Preprocessing...")
    processed_data = preprocess(data)
    print("Saving data...")
    np.save(TXT_FILEPATH.replace('.txt', '.npy'), processed_data)
    print("Generating sequences...")
    gen_sequences(processed_data)
    print('Done')

if __name__ == '__main__':
    main()
