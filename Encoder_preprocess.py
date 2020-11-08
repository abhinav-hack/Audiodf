import librosa
import librosa.display
from playsound import playsound
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import time

HOP_LENGTH = 256
SR = 22050
DATASET_PATH = "/home/hacker/Documents/audio/vcc2016_data"
N_MFCC = 20
SAMPLE_LEN = 22050 * 4


def encode_dataset(file_path, hop_length=256, n_mfcc=20, n_fft=2048, sr=22050):
    
    """
        prepare mfcc from the audio files """ 

    start = time.time()

    # processing audio

    voice_map, sr = librosa.load(file_path)

    if len(voice_map) < SAMPLE_LEN:
        voice_map = np.pad(voice_map, (0,SAMPLE_LEN-len(voice_map)))
    else :
        voice_map = voice_map[:SAMPLE_LEN]

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
    comprehensive_mfccs = comprehensive_mfccs[..., np.newaxis]
    # try chroma also - displays energy in nodes
    """ chromagram = librosa.feature.chroma_stft(signal, sr=sr, hop_length=HOP_LENGTH)
    plt.figure(figsize=(15, 5))
    librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=HOP_LENGTH, cmap='coolwarm')
    """
    
    stop = time.time()
    print(comprehensive_mfccs.shape)
    print("time :", stop-start)
    return comprehensive_mfccs

def access_file(dataset_path):
    """ Return all file path in the given directory"""

    file_list = []
    for dirpath, dirnames, filenames in os.walk(dataset_path):
        for f in filenames:
            file_path = os.path.join(dirpath, f)
            file_list.append(file_path)
            print(file_path)
    return file_list


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
        tm = encode_dataset(files)
        audio_list.append(tm)
    
""" 
    plot_spec(log_mfccs, sr, x_axis='time', y_axis='linear')
    plot_spec(log_delta, sr, x_axis='time', y_axis='linear')
    plot_spec(log_delta2, sr,  x_axis='time', y_axis='linear')
    print(log_mfccs.shape, log_delta.shape, log_delta2.shape)
    """