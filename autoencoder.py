import numpy as np

from keras.callbacks import EarlyStopping, ModelCheckpoint
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.layers import Conv2D
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

TXT_FILEPATH = r'C:\Users\booga\Dropbox\projects\argus\Argus_Protocol_Research\reference_recording_01.txt'
ENCODER_MODEL_FILEPATH = r'C:\Users\booga\Dropbox\projects\argus\Argus_Protocol_Research\encoder.h5'
ENCODING_DIM = 40
SLIDING_WINDOW_WIDTH = 20
data = np.load(TXT_FILEPATH.replace('.txt', '.npy'))
N, packet_features = data.shape


validation_ratio = 0.1
x_train = data[int(N * validation_ratio):, :]
x_validation = data[:int(N * validation_ratio), :]

input_layer = Input(shape=(packet_features,))
# shallow encoder

encoded = Dense(ENCODING_DIM, activity_regularizer=regularizers.l1(1e-4), activation='relu')(input_layer)
decoded = Dense(packet_features, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')
callbacks = [EarlyStopping(patience=5), ModelCheckpoint(ENCODER_MODEL_FILEPATH, save_best_only=True)]
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                callbacks=callbacks,
                validation_data=(x_validation, x_validation))
