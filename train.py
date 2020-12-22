import numpy as np
from playsound import playsound
import soundfile as sf
import librosa
from utils import *
from Encoder_preprocess import *
from Synthesizer_preprocess import *
from Combined_model import build_combined_model
import time

# build model combined
com_model = build_combined_model()
sr = 22050
EPOCHS = 1
# load audio files
sound_file_path_list = access_file("./vcc2016_data/SM1")

#load model weights
start = time.time()
for epoch in range(EPOCHS):
    #com_model.load_weights("./model_check/checkpoint")
    print("Epochs : ", epoch)
    for sound_file in sound_file_path_list:

        sound = load_audio(sound_file)

        # prepare encoder array
        enc_mfccs = encode_dataset(sound)

        # prepare synthesizer array
        syn_mel = synthesize_dataset(sound)

        # reshape and train
        sound = sound.reshape((345, 256))
        loss = com_model.fit(x=[enc_mfccs, syn_mel], y=sound, batch_size=1)
        print(com_model.metrics_names, loss)
        stop = time.time()
        print(stop-start)
    print("saving checkpoints, Do not stop training")
    com_model.save_weights("./model_check/checkpoint")
    print("done, checkpoints saved.")
stop = time.time()
print(stop-start)
