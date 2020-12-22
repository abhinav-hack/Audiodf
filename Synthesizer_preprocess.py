import librosa
import librosa.display
from playsound import playsound
import matplotlib.pyplot as plt
import numpy as np
import playsound
import os
import time
from utils import *



DATASET_PATH = "/home/hacker/Documents/audio/vcc2016_data"


def synthesize_dataset(signal, hop_length=256, n_mels=60, n_fft=2048, sr=22050):
    
    """
        prepare mel spectrogram from the audio files """ 



    mel = librosa.feature.melspectrogram(signal, sr=sr, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels)

    log_mel = librosa.power_to_db(mel)

    log_mel_trans = log_mel[..., np.newaxis].T


    log_mel_trans = log_mel_trans.reshape((BATCH, 1, 60))
    log_mel_trans = np.delete(log_mel_trans, BATCH-1, axis=0)   #(345, 1, 60)
    #log_mel_trans = log_mel_trans.reshape(((BATCH-1)//N, N, 60))  #(69, 5, 60)

    #print(log_mel_trans.shape)

    return log_mel_trans

def plot_spec(signal, sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel'):
                    plt.figure(figsize=(25, 10))
                    librosa.display.specshow(signal, 
                                            x_axis=x_axis,
                                            y_axis=y_axis, 
                                            hop_length=HOP_LENGTH,
                                            sr=sr)
                    plt.colorbar(format="%+2.f")
                    plt.show()
                    
if __name__ == "__main__":
    
#    synthesize_dataset(file_path="/home/hacker/Documents/audio/vcc2016_data/SF1/100086.wav")

    file_list = access_file("/home/hacker/Documents/audio/vcc2016_data/SF1/")
    sym_list = []
    for files in file_list:
        sym = load_audio(files)
        sym = synthesize_dataset(sym)
        sym_list.append(sym)

""" plot_spec(mel, sr)
    plot_spec(mel, sr, y_axis='linear')
    plot_spec(log_mel, sr)
    plot_spec(log_mel, sr, y_axis='linear')
    print(mel_transpose.shape)
    """
