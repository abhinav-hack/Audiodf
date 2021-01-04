from numba import jit, cuda
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Input, Activation
from keras.layers import BatchNormalization, concatenate
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam, SGD,Adadelta
from keras.losses import mse, kullback_leibler_divergence, mean_absolute_error, mean_squared_logarithmic_error
import soundfile as sf
from Encoder_model import build_encoder
from Synthesizer_model import build_synthesizer
from Vocoder_model import build_vocoder
from Synthesizer_preprocess import *
from Encoder_preprocess import *


def my_loss_fn(y, a):
    cross_ent = ((1+y)/2*tf.math.log(a))+((1-y)/2*tf.math.log(1-a))
    return tf.reduce_mean(cross_ent, axis=-1)


def build_combined_model():
    encoder = build_encoder()
    synthesizer = build_synthesizer()
    vocoder = build_vocoder()

    comb_lyr = concatenate([encoder.output, synthesizer.output])
    output_model = vocoder(comb_lyr)
    combined = Model(inputs=[encoder.input, synthesizer.input], outputs=output_model, name='combined')
    combined.compile(optimizer=SGD(learning_rate=0.1), loss='kullback_leibler_divergence', metrics=['accuracy'])
    keras.utils.plot_model(combined, show_shapes=True)
    combined.summary()
    return combined

if __name__ == "__main__":
    combined = build_combined_model()

