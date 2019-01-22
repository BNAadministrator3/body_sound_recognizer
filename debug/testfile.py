from librosa import feature as ft
import numpy as np
from general_func.file_wav import read_wav_data, GetFrequencyFeatures
import time

if __name__ == '__main__':
    path = "/home/zhaok14/example/PycharmProjects/setsail/individual_spp/bowelsounds/unbalanced/validation/bowels/244814983784728_2017_03_29_21_31_25.wav"
    wavsignal, fs = read_wav_data(path)
    start = time.time()
    feat1 = GetFrequencyFeatures(wavsignal,fs)
    print('soectrogram shape:',feat1.shape)
    melspek = ft.melspectrogram(S=np.rot90(feat1), n_mels=26)
    melspek = np.rot90(melspek,1,(1,0))
    end = time.time()
    print('conventional:',round(end - start,2),'s')
    print('shape:',melspek.shape)

    start = time.time()
    melspek = ft.melspectrogram(np.array(wavsignal[0],dtype=np.float32),fs, n_fft=400, hop_length=160, power=2.0, n_mels=26)
    melspek = np.rot90(melspek, 1, (1, 0))
    end = time.time()
    print('new:', round(end - start, 2), 's')
    print('shape:', melspek.shape)




