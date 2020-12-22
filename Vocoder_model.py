import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Input, Activation
from keras.layers import BatchNormalization
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from keras.losses import mse
import soundfile as sf
from Encoder_model import *
from Synthesizer_model import *


def build_vocoder():
    """
    building vocoder model
    """
    voc_inputs = Input(shape = (320,))
    dense_lyr = Dense(512, activation='relu')(voc_inputs)
    dense_lyr = Dense(512, activation='sigmoid')(dense_lyr)
    voc_output = Dense(256, activation='tanh')(dense_lyr)
    vocoder = Model(inputs=voc_inputs, outputs=voc_output, name = 'vocoder')
    vocoder.summary()

    # Compiling Vocoder

    vocoder.compile(optimizer=Adam(), loss=mse, metrics=['accuracy'])
    keras.utils.plot_model(vocoder, show_shapes= True)

    return vocoder
