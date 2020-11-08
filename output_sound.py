from numba import jit, cuda
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Input, Activation
from keras.layers import BatchNormalization, concatenate
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from keras.losses import mse
import soundfile as sf
from Encoder_model import build_encoder
from Synthesizer_model import build_synthesizer
from Vocoder_model import build_vocoder
from Synthesizer_preprocess import *
from Encoder_preprocess import *

encoder = build_encoder()
synthesizer = build_synthesizer()
vocoder = build_vocoder()

voice_map_input = Input(shape=(N_MFCC*3, 345, 1))
signal_input = Input(shape=(345, N_MELS))
comb_enc = encoder(voice_map_input)
comb_syn = synthesizer(signal_input)
comb_lyr = concatenate([comb_enc, comb_syn])
output_model = vocoder(comb_lyr)

combined = Model(inputs=[signal_input, voice_map_input], outputs=output_model, name='combined')
combined.summary()

keras.utils.plot_model(combined, show_shapes=True)
