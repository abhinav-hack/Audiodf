import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Input, Activation, LSTM, BatchNormalization 
from keras.models import Model, Sequential, load_model
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.losses import mse
from Synthesizer_preprocess import *
from utils import *


def build_synthesizer():
    """
    build model
    Input to the lstm layer must be three dimentional
    """

    syn_inputs = Input(shape=(1, N_MELS))
    lstm_lyr = LSTM(64, return_sequences=True)(syn_inputs)  # (input = (timestep, features))
    lstm_lyr = LSTM(64)(lstm_lyr)
    dense_lyr = Dense(128, activation='relu')(lstm_lyr)
    syn_outputs = Dense(256, activation='relu')(dense_lyr)

    synthesizer = Model(inputs=syn_inputs, outputs=syn_outputs, name='synthesizer')
    synthesizer.summary()
    keras.utils.plot_model(synthesizer, show_shapes=True)

    synthesizer.compile(optimizer=Adam(), loss=mse, metrics=['accuracy'])
    return synthesizer


if __name__ == '__main__':
    synthesizer = build_synthesizer()