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


def synthesize_dataset(signal, hop_length=HOP_LENGTH, n_mels=N_MELS, n_fft=N_FFT, sr=SR):
    
    """
        prepare mel spectrogram from the audio files """ 



    mel = librosa.feature.melspectrogram(signal, sr=sr, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels)

    log_mel = librosa.power_to_db(mel)
    # Chroma displays energy of each pitch class
    chromagram = librosa.feature.chroma_stft(signal, sr=sr, hop_length=hop_length)
    chromagram = np.delete(chromagram,  int(SAMPLE_LEN/hop_length), axis=1).T
    zero_crossings = librosa.zero_crossings(signal, pad=False)
    zero_crossings = zero_crossings.reshape((int(SAMPLE_LEN/hop_length), hop_length)).sum(axis=1)/hop_length
    zero_crossings = zero_crossings.reshape((int(SAMPLE_LEN/hop_length), 1))
    log_mel_trans = log_mel[..., np.newaxis].T
    
    chrom_zero = np.concatenate((chromagram, zero_crossings), axis=1)
    chrom_zero = chrom_zero.reshape((int(SAMPLE_LEN/hop_length), 1, -1))

    log_mel_trans = log_mel_trans.reshape((BATCH, 1, n_mels))
    log_mel_trans = np.delete(log_mel_trans, int(SAMPLE_LEN/hop_length), axis=0)   #(345, 1, 60)
    #log_mel_trans = log_mel_trans.reshape(((int(SAMPLE_LEN/hop_length))//N, N, 60))  #(69, 5, 60)

    log_mel_chrom = np.concatenate((log_mel_trans, chrom_zero), axis=2)
   
    print(log_mel_chrom.shape)

    return log_mel_chrom

def plot_spec(signal, sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel'):
                    plt.figure(figsize=(25, 10))
                    librosa.display.specshow(signal, 
                                            x_axis=x_axis,
                                            y_axis=y_axis, 
                                            hop_length=hop_length,
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

    #plt.figure(figsize=(15, 5))
    #librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=HOP_LENGTH, cmap='coolwarm')

"""     plot_spec(mel, sr)
    plot_spec(mel, sr, y_axis='linear')
    plot_spec(log_mel, sr)
    plot_spec(log_mel, sr, y_axis='linear')
    print(mel_transpose.shape) """

