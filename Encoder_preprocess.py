import librosa
import librosa.display
from playsound import playsound
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import time
from utils import *


DATASET_PATH = "/home/hacker/Documents/audio/vcc2016_data"


def encode_dataset(voice_map, hop_length=256, n_mfcc=20, n_fft=2048, sr=22050):
    
    """
        prepare mfcc from the audio files """ 


    # processing audio


    mfccs = librosa.feature.mfcc(voice_map, hop_length=hop_length, n_mfcc=n_mfcc, n_fft=n_fft, sr=sr)
    #print('shape of mfcc : ', mfccs.shape)

    log_mfccs = librosa.power_to_db(mfccs)

    ### First and second derivate of mfccs

    delta_mfcc = librosa.feature.delta(mfccs)
    delta2_mfcc = librosa.feature.delta(mfccs, order=2)

    log_delta = librosa.power_to_db(delta_mfcc)
    log_delta2 = librosa.power_to_db(delta2_mfcc)
    
    ### Concatenate all in one vector

    comprehensive_mfccs = np.concatenate((log_mfccs, 
                                        log_delta,
                                        log_delta2))    
    
    #comprehensive_mfccs = comprehensive_mfccs.reshape((60, BATCH))

    comprehensive_mfccs = np.tile(comprehensive_mfccs, ((BATCH-1), 1, 1)) #(345, 60, 346) 69-copies
    

    # try chroma also - displays energy in nodes
    """ chromagram = librosa.feature.chroma_stft(signal, sr=sr, hop_length=HOP_LENGTH)
    plt.figure(figsize=(15, 5))
    librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=HOP_LENGTH, cmap='coolwarm')
    """
    
    #print(comprehensive_mfccs.shape)
    return comprehensive_mfccs


def plot_spec(voice_map, sr, hop_length=HOP_LENGTH, x_axis=None, y_axis=None):
                    ''' 
                        function for plotting spectogram of given data
                    '''
                    plt.figure(figsize=(25,10))
                    librosa.display.specshow(voice_map, sr=sr,
                                            hop_length=HOP_LENGTH,
                                            x_axis=x_axis,
                                            y_axis= y_axis)
                    plt.colorbar(format="%+2.f")
                    plt.show()

if __name__ == "__main__":

    file_list = access_file("/home/hacker/Documents/audio/vcc2016_data/SF1/")
    audio_list = []
    for files in file_list:
        tm = load_audio(files)
        tm = encode_dataset(tm)
        audio_list.append(tm)
    
""" 
    plot_spec(log_mfccs, sr, x_axis='time', y_axis='linear')
    plot_spec(log_delta, sr, x_axis='time', y_axis='linear')
    plot_spec(log_delta2, sr,  x_axis='time', y_axis='linear')
    print(log_mfccs.shape, log_delta.shape, log_delta2.shape)
    """