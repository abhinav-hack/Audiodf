import os 
import librosa
import numpy as np


HOP_LENGTH = 256
SR = 22050
N_MFCC = 20
N_MELS = 60
SAMPLE_LEN = 88320   # taking round figure for model
BATCH = (SAMPLE_LEN // HOP_LENGTH) +1
N = 5
# load sound file with shape (69, 1280)


def access_file(dataset_path):
    """ Return list all file path in the given directory"""

    file_list = []
    try:
        for dirpath, dirnames, filenames in os.walk(dataset_path):
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                file_list.append(file_path)
                print(file_path)
        return file_list
    except :
        print("file not found for accessing dataset")
        


def load_audio(file_path):
    """ load audio with specific length and sr with librosa only for one file """
    try:
        signal, sr = librosa.load(file_path)
    
        if len(signal) < SAMPLE_LEN:
            signal = np.pad(signal, (0,SAMPLE_LEN-len(signal)))
        else :
            signal = signal[:SAMPLE_LEN]
        
        return signal
    except :
        print("Error reading audio file")
        