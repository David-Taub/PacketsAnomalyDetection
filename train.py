import numpy as np

from keras.callbacks import EarlyStopping, ModelCheckpoint
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.layers import Conv2D
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Flatten
from keras.layers import Input, Dense
from keras.layers import Dropout
from keras.models import Model
from keras import regularizers


TXT_FILEPATH = r'C:\Users\booga\Dropbox\projects\PacketsAnomalyDetection\Argus_Protocol_Research\reference_recording_01.txt'
ENCODER_MODEL_FILEPATH = r'C:\Users\booga\Dropbox\projects\PacketsAnomalyDetection\Argus_Protocol_Research\encoder.h5'
# X_TRAIN_MM_FILEPATH = r'C:\Users\booga\Dropbox\projects\PacketsAnomalyDetection\Argus_Protocol_Research\x_train.mm'
# Y_TRAIN_MM_FILEPATH = r'C:\Users\booga\Dropbox\projects\PacketsAnomalyDetection\Argus_Protocol_Research\y_train.mm'
# X_VALIDATION_MM_FILEPATH = r'C:\Users\booga\Dropbox\projects\PacketsAnomalyDetection\Argus_Protocol_Research\x_validation.mm'
# Y_VALIDATION_MM_FILEPATH = r'C:\Users\booga\Dropbox\projects\PacketsAnomalyDetection\Argus_Protocol_Research\y_validation.mm'
ENCODING_DIM = 60
SLIDING_WINDOW_WIDTH = 30


def generator(data, window_size, batch_size):
    features = data.shape[1]
    while True:
        for i in range(int(np.ceil((data.shape[0] - window_size) / batch_size))):
            y = data[i * batch_size + window_size - 1: (i + 1) * batch_size + window_size - 1, :]
            current_batch_size = y.shape[0]
            x = np.zeros((current_batch_size, window_size - 1, features))
            for j in range(current_batch_size):
                x[j, :, :] = data[i * batch_size + j: i * batch_size + j + window_size - 1, :]
            yield (x, y)


data = np.load(TXT_FILEPATH.replace('.txt', '.npy'))
packet_features = data.shape[1]
# TODO: code duplication, save it in json file in preprocessing
validation_ratio = 0.1
DROUPUT_RATE = 0.2
LSTM_SIZE = 50
train_size = int((data.shape[0] - SLIDING_WINDOW_WIDTH) * (1 - validation_ratio))

model = Sequential()

ENCODING_DIM = packet_features
#TODO: load weights from autoencoder
# model.add(Reshape((SLIDING_WINDOW_WIDTH - 1, packet_features, 1), input_shape=(SLIDING_WINDOW_WIDTH - 1, packet_features)))
# model.add(Conv2D(ENCODING_DIM, (1, packet_features), input_shape=(SLIDING_WINDOW_WIDTH - 1, packet_features, 1)))
# model.add(Reshape((SLIDING_WINDOW_WIDTH - 1, ENCODING_DIM), input_shape=(SLIDING_WINDOW_WIDTH - 1, ENCODING_DIM, 1)))

model.add(Bidirectional(LSTM(LSTM_SIZE, return_sequences=True), input_shape=(SLIDING_WINDOW_WIDTH - 1, ENCODING_DIM), merge_mode='concat', name='BiLSTM1'))
model.add(Dropout(DROUPUT_RATE, input_shape=(2, LSTM_SIZE), name='dropout1'))
model.add(Bidirectional(LSTM(LSTM_SIZE), input_shape=(1, LSTM_SIZE), merge_mode='concat', name='BiLSTM2'))
model.add(Dense(100, input_shape=(2 * LSTM_SIZE,), activation='relu', name='dense1'))
model.add(Dropout(DROUPUT_RATE, input_shape=(100,), name='dropout2'))
model.add(Dense(packet_features, input_shape=(100,), activation='sigmoid', name='dense2'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[metrics.binary_accuracy])
callbacks = [EarlyStopping(patience=10), ModelCheckpoint(ENCODER_MODEL_FILEPATH, save_best_only=True)]
# TODO: split the packet by enums, and use on each a softmax activation + crossentropy


batch_size = 64
model.fit_generator(generator(data[:train_size, :], SLIDING_WINDOW_WIDTH, batch_size),
                    steps_per_epoch=int(np.ceil(train_size / batch_size)),
                    epochs=50,
                    callbacks=callbacks,
                    validation_data=generator(data[train_size:, :], SLIDING_WINDOW_WIDTH, batch_size),
                    validation_steps=int(np.ceil((data.shape[0] - train_size) / batch_size)))

