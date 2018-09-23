import os
import sys
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import metrics
from keras.models import Sequential
# from keras.layers import Reshape
# from keras.layers import Conv2D
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import Dropout

dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))
regular_recording = os.path.join(dirname, r'reference_recording_01.txt')
model_filepath = os.path.join(dirname, r'deep_ML.h5')

EMBEDDING_DIM = 60
SLIDING_WINDOW_WIDTH = 30
VALIDATION_RATIO = 0.1
DROUPUT_RATE = 0.25
LSTM_SIZE = 50


def sequence_gen(data, window_size, batch_size):
    features = data.shape[1]
    while True:
        for i in range(int(np.ceil((data.shape[0] - window_size) / batch_size))):
            y = data[i * batch_size + window_size - 1: (i + 1) * batch_size + window_size - 1, :]
            current_batch_size = y.shape[0]
            x = np.zeros((current_batch_size, window_size - 1, features))
            for j in range(current_batch_size):
                x[j, :, :] = data[i * batch_size + j: i * batch_size + j + window_size - 1, :]
            yield (x, y)


data = np.load(regular_recording.replace('.txt', '.npy'))
packet_features = data.shape[1]
# TODO: code duplication, save it in json file in preprocessing

model = Sequential()

#if no word2vec style embedding, the original features are the encoded features
EMBEDDING_DIM = packet_features

#TODO: consider load weights from a pretrained autoencoder
# model.add(Reshape((SLIDING_WINDOW_WIDTH - 1, packet_features, 1), input_shape=(SLIDING_WINDOW_WIDTH - 1, packet_features)))
train_size = int((data.shape[0] - SLIDING_WINDOW_WIDTH) * (1 - VALIDATION_RATIO))
# model.add(Conv2D(EMBEDDING_DIM, (1, packet_features), input_shape=(SLIDING_WINDOW_WIDTH - 1, packet_features, 1)))
# model.add(Reshape((SLIDING_WINDOW_WIDTH - 1, EMBEDDING_DIM), input_shape=(SLIDING_WINDOW_WIDTH - 1, EMBEDDING_DIM, 1)))

model.add(Bidirectional(LSTM(LSTM_SIZE, return_sequences=True),
                        input_shape=(SLIDING_WINDOW_WIDTH - 1, EMBEDDING_DIM),
                        merge_mode='concat', name='BiLSTM1'))
model.add(Dropout(DROUPUT_RATE, input_shape=(2, LSTM_SIZE), name='dropout1'))
model.add(Bidirectional(LSTM(LSTM_SIZE), input_shape=(1, LSTM_SIZE), merge_mode='concat', name='BiLSTM2'))
model.add(Dropout(DROUPUT_RATE, input_shape=(2 * LSTM_SIZE,), name='dropout2'))
model.add(Dense(100, input_shape=(2 * LSTM_SIZE,), activation='relu', name='dense1'))
model.add(Dropout(DROUPUT_RATE, input_shape=(100,), name='dropout3'))
model.add(Dense(packet_features, input_shape=(100,), activation='sigmoid', name='dense2'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[metrics.binary_accuracy])

# TODO: split the packet by enums, and use on each a softmax activation + crossentropy
# TODO: try adding regularization


batch_size = 64
callbacks = [EarlyStopping(patience=5), ModelCheckpoint(model_filepath, save_best_only=True)]
model.fit_generator(sequence_gen(data[:train_size, :], SLIDING_WINDOW_WIDTH, batch_size),
                    steps_per_epoch=int(np.ceil(train_size / batch_size)),
                    epochs=50,
                    callbacks=callbacks,
                    validation_data=sequence_gen(data[train_size:, :], SLIDING_WINDOW_WIDTH, batch_size),
                    validation_steps=int(np.ceil((data.shape[0] - train_size) / batch_size)))

