#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import platform as plat
import os
import keras
import matplotlib as plt

import numpy as np
from general_func.file_wav import *
from general_func.file_dict import *
from help_func.get_interval_index import get_interval_list

import random

MAX_AUDIO_LENGTH = 110
AUDIO_FEATURE_LENGTH = 200
CLASS_NUM = 4
FOLDER_SPLIT_NUM = 5

class DataCross():

    def __init__(self, path, LoadToMem=False, MemWavCount=10000):

        '''
        参数：
            path：数据存放位置根目录
            type: to distinguish the 4-1 folder
        '''
        system_type = plat.system()  # 由于不同的系统的文件路径表示不一样，需要进行判断
        self.datapath = path  # 数据存放位置根目录

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

        self.folds = []
        self.__LoadDataList()

        self.list_normal = []
        self.list_wheeze = []
        self.list_crackle = []
        self.list_both = []
        self.list_all = []
        self.class_num = CLASS_NUM
        self.DataNum = 0

        self.feat_dimension = AUDIO_FEATURE_LENGTH
        self.frame_length = 400

        pass

    def __LoadDataList(self):
        '''
        加载用于计算的数据列表
        参数：
        '''
        # the classic codes to browse all the subfolders.
        folder_list= []
        if (os.path.exists(self.datapath)):
            directory = os.listdir(self.datapath)
            for name in directory:
                m = os.path.join(self.datapath,name)
                if os.path.isdir(m):
                    folder_list.append(m)

        if FOLDER_SPLIT_NUM!=len(folder_list):
            print('*[Error] Dataset misplacement!')
            assert(0)
        else:
            try:
                for fpath in folder_list:
                    normal_path = os.path.join(fpath,'normal')
                    normal_list = os.listdir(normal_path)
                    normal_list = [os.path.join(normal_path, element) for element in normal_list ]
                    wheeze_path = os.path.join(fpath, 'wheeze')
                    wheeze_list = os.listdir(wheeze_path)
                    wheeze_list = [os.path.join(wheeze_path, element) for element in wheeze_list ]
                    crackle_path = os.path.join(fpath, 'crackle')
                    crackle_list = os.listdir(crackle_path)
                    crackle_list = [os.path.join(crackle_path,element) for element in crackle_list]
                    # print(len(crackle_list))
                    both_path = os.path.join(fpath, 'both')
                    both_list = os.listdir(both_path)
                    both_list = [os.path.join(both_path, element) for element in both_list ]
                    tmp_dict = {'normal':normal_list,'wheeze':wheeze_list,'crackle':crackle_list,'both':both_list}
                    self.folds.append(tmp_dict)
            except:
                print('*[Error] Dataset misformation!')
                assert (0)

    def SplitType(self, type = 'train', order = 1):
        ''''
        parameter: order belongs to [0, FOLDERSPLIT_NUM)

        '''
        #empty the lists
        self.list_normal = []
        self.list_wheeze = []
        self.list_crackle = []
        self.list_both = []
        self.list_all = []

        if order>=FOLDER_SPLIT_NUM:
            print('index out of the boundry.')
            assert(0)

        random.seed(datetime.datetime.now())
        if type == 'train':
            for i in [ x for x in range(FOLDER_SPLIT_NUM) if x!=int(order)]:
                self.list_normal = self.list_normal + self.folds[i]['normal']
                self.list_wheeze = self.list_wheeze + self.folds[i]['wheeze']
                self.list_crackle = self.list_crackle + self.folds[i]['crackle']
                self.list_both = self.list_both + self.folds[i]['both']
            self.list_all = self.list_all + [{element:'normal'} for element in self.list_normal] \
                                          + [{element: 'wheeze'} for element in self.list_wheeze] \
                                          + [{element: 'crackle'} for element in self.list_crackle] \
                                          + [{element: 'both'} for element in self.list_both]
            self.DataNum = (len(self.list_normal) , len(self.list_wheeze) , len(self.list_crackle) , len(self.list_both))
            random.shuffle(self.list_normal)
            random.shuffle(self.list_wheeze)
            random.shuffle(self.list_crackle)
            random.shuffle(self.list_both)
        elif type  == 'eval':
            i = int(order)
            for n in self.folds[i]['normal']:
                self.list_normal.append({n:'normal'})
            for w in self.folds[i]['wheeze']:
                self.list_wheeze.append({w:'wheeze'})
            for c in self.folds[i]['crackle']:
                self.list_crackle.append({c:'crackle'})
            for b in self.folds[i]['both']:
                self.list_both.append({b:'both'})
            self.list_all = self.list_both + self.list_normal + self.list_crackle + self.list_wheeze
            self.DataNum = len(self.list_all)
            del self.list_normal, self.list_wheeze, self.list_crackle, self.list_both
        else:
            print('incorrect parameters.')
            assert(0)
        random.shuffle(self.list_all)

    def GetDataEval(self, n_start, n_amount=1, mode='balanced'):
        '''
        :param n_start:
        :param n_amount:
        :param mode:
        :return:
        :note:  cycles less than 1s are also inclusive in the testing phase.
        '''

        # category = (self.list_normal, self.list_wheeze, self.list_crackle, self.list_both)
        # link = ('00', '01', '10', '11')
        # label = (0, 1, 2, 3)
        char2digit = {'normal': 0, 'wheeze': 1, 'crackle': 2, 'both': 3}
        data_label = []

        # while True:
        #     path = ''
        #     if mode == 'balanced':
        #         print('balanced mode is not available.')
        #     if mode == 'non-repetitive':
        #         path = list(self.list_all[n_start].keys)[0]
        #         str = list(self.list_all[n_start].keys)[0]
        #         try:
        #             data_label = np.array([char2digit[str]])
        #         except:
        #             print('label errors.')
        #             assert(0)
        #     wavsignal, fs = read_wav_data(path)
        #     if (wavsignal.shape[1] / fs > 1.0):
        #         break

        path = ''
        if mode == 'balanced':
            print('balanced mode is not available.')
            assert(0)
        if mode == 'non-repetitive':
            path = list(self.list_all[n_start].keys())[0]
            str = list(self.list_all[n_start].values())[0]
            try:
                data_label = np.array([char2digit[str]])
            except:
                print('*[ERROR] label errors.')
                assert (0)
        wavsignal, fs = read_wav_data(path)

        data_input = GetFrequencyFeatures(wavsignal, fs, self.feat_dimension, self.frame_length)
        data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)

        return data_input, data_label

    def __GetData4tTrain(self, n_start, n_amount=1):
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
        category = (self.list_normal, self.list_wheeze, self.list_crackle, self.list_both)
        label = (0, 1, 2, 3)
        data_label = []

        data_input = np.zeros(shape=(4, MAX_AUDIO_LENGTH, self.feat_dimension, 1), dtype=np.float)
        data_label = np.zeros(shape=(4, 1), dtype=np.int16)
        path = ''
        for p in range(len(category)):
            path = list(category[p][n_start % len(category[p])].keys())[0]
            data_label[p] = np.array([label[p]])
            wavsignal, fs = read_wav_data(path)
            datum_input = GetFrequencyFeatures(wavsignal, fs, self.feat_dimension, self.frame_length)
            if len(datum_input)<= MAX_AUDIO_LENGTH:
                pass
            else:
                span = len(datum_input) - MAX_AUDIO_LENGTH
                begin = np.random.randint(span)
                datum_input = datum_input[begin:begin+MAX_AUDIO_LENGTH]
            data_input[p, 0:len(datum_input)] = datum_input.reshape(datum_input.shape[0], datum_input.shape[1], 1)

        return data_input, data_label

    def data_genetator4tTrain(self, batch_size=32):
        '''
        数据生成器函数，用于Keras的generator_fit训练
        batch_size: 一次产生的数据量
        '''

        iterations_per_epoch = max(self.DataNum) // (batch_size // 4) + 1
        while True:
            ran_num = random.randint(0, iterations_per_epoch - 1) * (batch_size // 4)  # 获取discrete随机数
            assert (batch_size % 4 == 0)
            X = np.zeros((batch_size, MAX_AUDIO_LENGTH, self.feat_dimension, 1), dtype=np.float)
            y = np.zeros((batch_size, 1), dtype=np.int16)
            for i in range(batch_size // 4):
                data_input, data_label = self.__GetData4tTrain(n_start=(ran_num + i))  # 从随机数开始连续向后取一定数量数据
                lp = i * 4  # loading point
                X[lp:lp + 4] = data_input
                y[lp:lp + 4] = data_label
            yield [X, keras.utils.to_categorical(y, num_classes=self.class_num)], keras.utils.to_categorical(y,
                                                                                                             num_classes=self.class_num)  # 功能只是转成独热编码
        pass

    def StatisticData(self):
        length = []
        for i in range(sum(self.DataNum)):
            data_input, data_labels = self.GetDataEval(i, mode='non-repetitive')
            length.append(data_input.shape[0])
        print('mean value:',np.mean(length))
        binwidth = 20
        plt.hist(length, bins=np.arange(min(length), max(length) + binwidth, binwidth))
        plt.show()


if (__name__ == '__main__'):
    path = '/home/zhaok14/example/PycharmProjects/setsail/individual_spp/dataset/5-folds'
    l = DataCross(path)
    # l.SplitType(type='train',order=2)
    # print(l.folds)
    str = ['train', 'eval']
    l5 = []
    for i in range(5):
        lst = []
        for s in str:
            l.SplitType(type=s,order=i)
            print('{} order cross validation, {} set: '.format(i,s))
            print('data amounts:',l.DataNum)
            if s == 'train':
                print('overall sum',sum(l.DataNum))
                lst.append(sum(l.DataNum))
            if s == 'eval':
                print('sum unnecessarily.')
                lst.append(l.DataNum)
        l5.append(lst[0]+lst[1])
    print('check if mutually identical:',l5)