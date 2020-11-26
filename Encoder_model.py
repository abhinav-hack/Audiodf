import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, LSTM
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.losses import mse
from keras.layers import concatenate, Flatten, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from Encoder_preprocess import *
from utils import *



def build_encoder():
    """
    Creating Model  
    """

    enc_inputs = Input(shape=(N_MFCC*3, BATCH, 1))
    conv_lyr = Conv2D(64, kernel_size=3, padding='same', activation='relu')(enc_inputs)
    pooled_lyr = MaxPooling2D(pool_size=2, padding='same')(conv_lyr)
    conv_lyr = Conv2D(32, kernel_size=3, padding='same', activation='relu')(pooled_lyr)
    flat_lyr = Flatten()(conv_lyr)
    dense_lyr = Dense(128, activation=LeakyReLU(alpha=0.1))(flat_lyr)
    enc_outputs = Dense(64)(dense_lyr)

    encoder = Model(inputs=enc_inputs, outputs=enc_outputs, name="encoder")
    encoder.summary()

    # Compiling Model

    encoder.compile(optimizer=Adam(), loss=mse, metrics=['accuracy']) 

    keras.utils.plot_model(encoder, show_shapes=True)

    return encoder

if __name__ == "__main__" :
    build_encoder()