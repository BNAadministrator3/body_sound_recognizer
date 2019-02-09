from python_speech_features import mfcc
# import scipy.io.wavfile as wav
# import matplotlib.pyplot as plt
import numpy as np
from general_func.file_wav import read_wav_data, GetFrequencyFeatures
from multiprocessing import Pool
import time
from scipy import stats

# the argumens is infact not working.
# not pass the test.
def mfccFeatures(wave_data, samplerate, featurelength=200, framelength=400):
    '''
    将音频信号转化为帧。
    参数含义：
    wave_data:原始音频型号
    nw:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
    inc:相邻帧的间隔（同上定义）
    '''
    framestep = 100
    wlen=wave_data.shape[1] #信号总长度
    if wlen<=framelength: #若信号长度小于一个帧的长度，则帧数定义为1
        nf=1
    else: #否则，计算帧的总长度
        nf= np.floor((wlen - framelength)/framestep)+1
    picture = np.zeros(shape=(int(nf),int(featurelength)),dtype=np.float32)
    for i in range(int(nf)):
        offset = framestep * i
        frame = wave_data[0][0+offset:framelength+offset]
        temp = mfcc(frame,samplerate=4000,winlen=0.01,winstep=0.01,numcep=20,appendEnergy=False)
        if np.reshape(temp,(1,-1)).shape[1]!=200:
            print('frame number:',i)
            print( '??',np.reshape(temp,(1,-1)).shape[1] )
            assert(0)
        picture[i,:] = np.reshape(temp,(1,-1))

    return picture   #返回帧信号矩阵

def SimpleMfccFeatures(wave_data, samplerate, featurelength=26):
    temp = mfcc(wave_data[0], samplerate=samplerate, winlen=0.1, winstep=0.04, numcep=featurelength, appendEnergy=False)
    temp = temp[0:123, :]
    # return stats.zscore(temp)
    b = (temp - np.min(temp)) / np.ptp(temp)
    return b

def singleMfcc(frame):
    temp = mfcc(frame, samplerate=4000, winlen=0.01, winstep=0.01, numcep=20, appendEnergy=False)
    if np.reshape(temp, (1, -1)).shape[1] != 200:
        print('[Error]: unexpected size.')
        print('??', np.reshape(temp, (1, -1)).shape[1])
        assert (0)
    return np.reshape(temp,(1,-1))

#note the checking result is false for this function.
def mfccFeaturesMultithreads(wave_data, samplerate, featurelength=200, framelength=400):
    framestep = 100
    wlen = wave_data.shape[1]  # 信号总长度
    if wlen <= framelength:  # 若信号长度小于一个帧的长度，则帧数定义为1
        nf = 1
    else:  # 否则，计算帧的总长度
        nf = np.floor((wlen - framelength) / framestep) + 1
    frames = []
    for i in range(int(nf)):
        offset = framestep * i
        frame = wave_data[0][0 + offset:framelength + offset]
        frames.append(frame)
    pool = Pool(20)
    image = pool.map(singleMfcc, frames)
    pool.close()
    pool.join()
    image = np.array(image)

    return image



if __name__ == '__main__':
    path = "/home/zhaok14/example/PycharmProjects/setsail/individual_spp/bowelsounds/unbalanced/validation/bowels/244814983784728_2017_03_29_21_31_25.wav"
    wavsignal, fs = read_wav_data(path)
    start = time.time()
    mfcc_feat1 = SimpleMfccFeatures(wavsignal,fs)
    end = time.time()
    print('conventional:',round(end - start,2),'s')
    print('shape:',mfcc_feat1.shape)

    # start = time.time()
    # mfcc_feat2 = mfccFeaturesMultithreads(wavsignal,fs)
    # end = time.time()
    # print('multithreads:', round(end - start, 2), 's')
    # print((mfcc_feat1==mfcc_feat2).all())
    #
    # start = time.time()
    # mfcc_feat3 =  GetFrequencyFeatures(wavsignal, fs)
    # end = time.time()
    # print('spectrogram:', round(end - start, 2), 's')

    # plt.plot(mfcc_feat)
    # plt.show()
