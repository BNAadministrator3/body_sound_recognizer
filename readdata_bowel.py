#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import platform as plat
import os
import keras

import numpy as np
from general_func.file_wav import *
from general_func.file_dict import *
from help_func.get_interval_index import get_interval_list

import random

MAX_AUDIO_LENGTH = 650  #size:200*197
AUDIO_FEATURE_LENGTH = 200
CLASS_NUM = 2


class DataSpeech():

    def __init__(self, path, type, LoadToMem=False, MemWavCount=10000):
        '''
        参数：
            path：数据存放位置根目录
        '''
        system_type = plat.system()  # 由于不同的系统的文件路径表示不一样，需要进行判断
        self.datapath = path  # 数据存放位置根目录
        self.type = type  # 数据类型，分为两种：训练集(train)、验证集(validation)

        self.slash = ''
        if (system_type == 'Windows'):
            self.slash = '\\'  # 反斜杠
        elif (system_type == 'Linux'):
            self.slash = '/'  # 正斜杠
        else:
            print('*[Warning] Unknown System\n')
            self.slash = '/'  # 正斜杠

        if (self.slash != self.datapath[-1]):  # 在目录路径末尾增加斜杠
            self.datapath = self.datapath + self.slash

        self.common_path = ''
        self.list_bowel1 = []
        self.list_non0 = []
        self.DataNum = ()  # 记录数据量
        self.LoadDataList()

        self.list_path = self.GenAll(self.type)

        self.class_num = CLASS_NUM

        self.feat_dimension = AUDIO_FEATURE_LENGTH
        self.frame_length = 400
        # self.max_time_step = self.GetMaxTimeStep()
        # self.min_time_step = self.GetMinTimeStep()

        pass

    def LoadDataList(self):
        '''
        加载用于计算的数据列表
        参数：
            type：选取的数据集类型
                train 训练集
                eval 验证集
        '''
        # 设定选取哪一项作为要使用的数据集

        if (self.type == 'train'):
            self.common_path = self.datapath + 'train' + self.slash
        elif (self.type == 'eval'):
            self.common_path = self.datapath + 'validation' + self.slash
        else:
            print('*[Error] Index reading error!\n')
            assert (0)
        self.list_bowel1 = os.listdir(self.common_path + 'bowels')
        self.list_non0 = os.listdir(self.common_path + 'non')
        self.DataNum = (len(self.list_bowel1), len(self.list_non0))

    def GenAll(self, type):

        s = []
        link = ('bowels','non')
        for i in link:
            list_name_folder = os.listdir(self.common_path + i)
            tag = 1 if i == 'bowels' else 0
            for j in list_name_folder:
                str = self.common_path + i + self.slash + j
                s.append((str,tag))
        random.shuffle(s)

        return s

    def shifting(self,image,bias=39):
        assert(int(bias)<=int(image.shape[0] *0.2))
        translation = random.randint(0,(bias-1)//3)*3
        case = random.randint(1,3)
        if case != 2:#up blank
            image[0:translation]=0
        if case != 1:
            image[-1-translation:-1] = 0
        return image

    def GetData(self, n_start, n_amount=1, mode='balanced'):
        '''
        读取数据，返回神经网络输入值和输出值矩阵(可直接用于神经网络训练的那种)
        参数：
            n_start：从编号为n_start数据开始选取数据
            n_amount：选取的数据数量，默认为1，即一次一个wav文件
        返回：
            四个音频四个label，
        '''
        # 随机从四个文件夹中拿一条数据，判断是否大于1s，否就重拿
        category = (self.list_bowel1, self.list_non0)
        link = ('bowels', 'non')
        label = (1, 0)

        data_label = []

        while True:
            path = ''
            if mode == 'balanced':
                ran_num = random.randint(0, 1)
                filename = category[ran_num][n_start % len(category[ran_num])]
                path = self.common_path + link[ran_num] + self.slash + filename
                data_label = np.array([label[ran_num]])
            if mode == 'non-repetitive':
                path = self.list_path[n_start][0]
                data_label = np.array([self.list_path[n_start][1]])
            wavsignal, fs = read_wav_data(path)
            if (wavsignal.shape[1] / fs > 1.0):
                break

        data_input = GetFrequencyFeatures(wavsignal, fs, self.feat_dimension, self.frame_length)
        if data_label[0] == 0:
            data_input = self.shifting(data_input)
        data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)

        return data_input, data_label

    def data_genetator(self, batch_size=32, audio_length=MAX_AUDIO_LENGTH):
        '''
        数据生成器函数，用于Keras的generator_fit训练
        batch_size: 一次产生的数据量
        '''

        while True:
            ran_num = random.randint(0, max(self.DataNum) - 1)  # 获取一个随机数

            X = np.zeros((batch_size, audio_length, self.feat_dimension, 1), dtype=np.float)
            y = np.zeros((batch_size, 1), dtype=np.int16)
            for i in range(batch_size):
                data_input, data_labels = self.GetData(n_start=(ran_num + i))  # 从随机数开始连续向后取一定数量数据
                X[i, 0:len(data_input)] = data_input
                y[i, 0:len(data_labels)] = data_labels
            yield [X, keras.utils.to_categorical(y, num_classes=self.class_num)], keras.utils.to_categorical(y,
                                                                                                                 num_classes=self.class_num)  # 功能只是转成独热编码
        pass

    def GetData4t(self, n_start, n_amount=1):
        '''
        读取数据，返回神经网络输入值和输出值矩阵(可直接用于神经网络训练的那种)
        参数：
            n_start：从编号为n_start数据开始选取数据
            n_amount：选取的数据数量，默认为1，即一次一个wav文件
        返回：
            四个音频四个label，
        note:
            data less than 1s are not deleted.
        '''
        # 随机从四个文件夹中拿一条数据，判断是否大于1s，否就重拿
        category = (self.list_healthy00, self.list_wheeze01, self.list_crackle10, self.list_both11)
        link = ('00', '01', '10', '11')
        label = (0, 1, 2, 3)
        char2digit = {'00': 0, '01': 1, '10': 2, '11': 3}
        data_label = []

        data_input = np.zeros(shape=(4, MAX_AUDIO_LENGTH, self.feat_dimension, 1), dtype=np.float)
        data_label = np.zeros(shape=(4, 1), dtype=np.int16)
        path = ''
        for p in range(len(category)):
            filename = category[p][n_start % len(category[p])]
            path = self.common_path + link[p] + self.slash + filename
            data_label[p] = np.array([label[p]])
            wavsignal, fs = read_wav_data(path)
            datum_input = GetFrequencyFeatures(wavsignal, fs, self.feat_dimension, self.frame_length)
            data_input[p, 0:len(datum_input)] = datum_input.reshape(datum_input.shape[0], datum_input.shape[1], 1)

        return data_input, data_label

    def data_genetator4t(self, batch_size=32):
        '''
        数据生成器函数，用于Keras的generator_fit训练
        batch_size: 一次产生的数据量
        '''
        random.seed(datetime.datetime.now())
        random.shuffle(self.list_healthy00)
        random.shuffle(self.list_wheeze01)
        random.shuffle(self.list_crackle10)
        random.shuffle(self.list_both11)
        iterations_per_epoch = max(self.DataNum) // (batch_size // 4) + 1
        while True:
            ran_num = random.randint(0, iterations_per_epoch - 1) * (batch_size // 4)  # 获取discrete随机数
            assert (batch_size % 4 == 0)
            X = np.zeros((batch_size, MAX_AUDIO_LENGTH, self.feat_dimension, 1), dtype=np.float)
            y = np.zeros((batch_size, 1), dtype=np.int16)
            for i in range(batch_size // 4):
                data_input, data_label = self.GetData4t(n_start=(ran_num + i))  # 从随机数开始连续向后取一定数量数据
                lp = i * 4  # loading point
                X[lp:lp + 4] = data_input
                y[lp:lp + 4] = data_label
            yield [X, keras.utils.to_categorical(y, num_classes=self.class_num)], keras.utils.to_categorical(y,
                                                                                                             num_classes=self.class_num)  # 功能只是转成独热编码
        pass

    def GetMaxTimeStep(self, ):
        max_value = 0
        for i in range(sum(self.DataNum)):
            data_input, data_labels = self.GetData(i, mode='non-repetitive')
            temp = data_input.shape[0]
            if (temp > max_value):
                max_value = temp
        return max_value

    def GetMinTimeStep(self, ):
        datain, _ = self.GetData(0)
        min_value = datain.shape[0]
        for i in range(sum(self.DataNum)):
            data_input, data_labels = self.GetData(i, mode='non-repetitive')
            temp = data_input.shape[0]
            if (temp < min_value):
                min_value = temp
        return min_value


if (__name__ == '__main__'):
    path = '/home/zhaok14/example/PycharmProjects/setsail/individual_spp/bowelsounds'
    l = DataSpeech(path, 'train')
    l.LoadDataList()
    # print('max time step:', l.max_time_step)
    # print('max time step:', l.GetMaxTimeStep())
    # print('min time step:', l.min_time_step)
    # print('min time step:', l.GetMinTimeStep())
    # print('data size:', l.DataNum)
    print(l.GetData(random.randint(0,8000), mode='non-repetitive'))
    aa = l.data_genetator(batch_size=1)
    for i in aa:
        a, b = i
        print(a, b)
    pass
