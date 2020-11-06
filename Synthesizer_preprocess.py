import librosa
import librosa.display
from playsound import playsound
import matplotlib.pyplot as plt
import numpy as np
import playsound
import os
import time

HOP_LENGTH = 256
SR = 22050
DATASET_PATH = "/home/hacker/Documents/audio/vcc2016_data"
N_MELS = 60


def prepare_dataset(dataset_path, hop_length=256, n_mels=N_MELS, n_fft=2048, sr=22050):
    
    """
        prepare mel spectrogram from the audio files """ 

    start = time.time()

    
    # loop through all the files 
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        if i > 1 :
            break

        
        if dirpath is not dataset_path:
            
            for f in filenames:
                file_path = os.path.join(dirpath, f)


                signal, sr = librosa.load(file_path)


                mel = librosa.feature.melspectrogram(signal, sr=sr, hop_length=HOP_LENGTH, n_fft=2048, n_mels=n_mels)

                log_mel = librosa.power_to_db(mel)


                print(log_mel.shape)
                
                
    stop = time.time()
    print("time :",stop-start)

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
    
    prepare_dataset(DATASET_PATH, hop_length=HOP_LENGTH, n_mels=N_MELS)

""" plot_spec(mel, sr)
    plot_spec(mel, sr, y_axis='linear')
    plot_spec(log_mel, sr)
    plot_spec(log_mel, sr, y_axis='linear')
    print(mel_transpose.shape)
    """