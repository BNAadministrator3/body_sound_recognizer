#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import platform as plat
import os
import random
import time
import numpy as np
from tqdm import tqdm

import tensorflow as tf
import keras.backend as k
import keras
from keras.layers import *
from keras import optimizers
from keras.models import Model

from general_func.file_wav import GetFrequencyFeatures, read_wav_data
from general_func.gen_func import Compare2
from debug.mfcc_trial import SimpleMfccFeatures
from help_func.utilities_keras import focal_loss, ReguBlock, block, XcepBlock
from help_func.utilties import plot_confusion_matrix

AUDIO_LENGTH = 123  #size:200*197
CLASS_NUM = 2
def clrdir(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            clrdir(c_path)
        else:
            os.remove(c_path)

def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        clrdir(path)
        print(path + ' 目录已存在,已清空里面内容')
        return False

def stringCheck(feature_type, module_type, layer_numbers):
    if feature_type in ('spec', 'Spec', 'SPEC', 'Spectrogram', 'SPECTROGRAM'):
        new_feature_type = 'spec'.upper()
    elif feature_type in ('mfcc', 'Mfcc', 'MFCC', 'MFC', 'mfc', 'Mfc'):
        new_feature_type = 'mfcc'.upper()
    else:
        print('*[ERROR]Unknown feature type.')
        assert (0)

    if module_type in ('regular', 'residual', 'inception'):
        new_module_type = module_type.capitalize()
    else:
        print('*[ERROR]Out of set: module_type.')
        assert (0)

    if layer_numbers in (2, 4, 6, 8, 10):
        new_layer_counts = layer_numbers
    else:
        print('*[ERROR]Out of set: layer_counts.')
        assert (0)
    return new_feature_type,new_module_type,new_layer_counts

class pathpoper():
    def __init__(self):
        base = os.path.split(os.path.realpath(__file__))[0]
        self.root = os.path.join(base, 'CNNdesign')
        child_mreg = 'mfcc+regular'
        child_sreg = 'spec+regular'
        child_mres = 'mfcc+residual'
        child_sres = 'spec+residual'
        child_mi = 'mfcc+inception'
        child_si = 'spec+inception'
        subfolders = (child_mreg, child_sreg, child_mres, child_sres, child_mi, child_si)
        for subfolder in subfolders:
            for layers in ('2', '4', '6', '8', '10'):
                mkdir(os.path.join(self.root, subfolder, layers))

    def popup(self,feature_type,module_type,layers):
        return os.path.join(self.root,feature_type.lower()+'+'+module_type.lower(),str(layers))

class DataSpeech():
    def __init__(self, path, feature_type, type):
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

        self.feature_type = feature_type
        self.feat_dimension = 200 if self.feature_type in ['spec','Spec','SPEC','Spectrogram','SPECTROGRAM'] else 26

        self.common_path = ''
        self.list_bowel1 = []
        self.list_non0 = []
        self.DataNum = ()  # 记录数据量
        self.LoadDataList()

        self.list_path = self.GenAll(self.type)

        self.class_num = CLASS_NUM
        self.frame_length = 400
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
        self.DataNum = (len(self.list_bowel1), len(self.list_non0)) #primary map

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

    def listShuffle(self,terminus):
        temp = self.list_bowel1[0:terminus]
        random.shuffle(temp)
        self.list_bowel1[0:terminus] = temp
        temp = self.list_non0[0:terminus]
        random.shuffle(temp)
        self.list_non0[0:terminus] = temp


    def shifting(self,image,bias=39):
        bias=int(image.shape[0] *0.2)
        translation = random.randint(0,(bias-1)//3)*3
        case = random.randint(1,3)
        if case != 2:#up blank
            image[0:translation]=0
        if case != 1:
            image[-1-translation:-1] = 0
        return image

    def GetData(self, n_start, n_amount=32,  mode='balanced'):
        '''
        读取数据，返回神经网络输入值和输出值矩阵(可直接用于神经网络训练的那种)
        参数：
            n_start：从编号为n_start数据开始选取数据
            n_amount：选取的数据数量，默认为1，即一次一个wav文件
        返回：
            四个音频四个label，
        '''
        # 随机从四个文件夹中拿一条数据，判断是否大于1s，否就重拿
        assert(n_amount%CLASS_NUM==0)
        category = (self.list_bowel1, self.list_non0)
        link = ('bowels', 'non')
        label = (1, 0)
        if self.feature_type in ['spec','Spec','SPEC','Spectrogram','SPECTROGRAM']:
            if mode == 'balanced':
                data = []
                labels = []
                for genre in range(CLASS_NUM):
                    for file in range(n_amount//CLASS_NUM):
                        filename = category[genre][(n_start + file)%self.DataNum[genre]]
                        # filename = category[genre][(n_start + file) % min(self.DataNum)]
                        path = self.common_path + link[genre] + self.slash + filename
                        wavsignal, fs = read_wav_data(path)
                        # data_input = SimpleMfccFeatures(wavsignal, fs)
                        data_input = GetFrequencyFeatures(wavsignal, fs, self.feat_dimension, self.frame_length,shift=160)
                        data_input = self.shifting(data_input)
                        data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)
                        data.append(data_input)
                        data_label = np.array([label[genre]])
                        labels.append(data_label)
                return data, labels
            if mode == 'non-repetitive':
                path = self.list_path[n_start][0]
                data_label = np.array([self.list_path[n_start][1]])
                wavsignal, fs = read_wav_data(path)
                data_input = GetFrequencyFeatures(wavsignal, fs, self.feat_dimension, self.frame_length,shift=160)
                data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)
                return data_input,  data_label
        elif self.feature_type in ['mfcc','MFCC','Mfcc']:
            if mode == 'balanced':
                data = []
                labels = []
                for genre in range(CLASS_NUM):
                    for file in range(n_amount//CLASS_NUM):
                        filename = category[genre][(n_start + file)%self.DataNum[genre]]
                        # filename = category[genre][(n_start + file) % min(self.DataNum)]
                        path = self.common_path + link[genre] + self.slash + filename
                        wavsignal, fs = read_wav_data(path)
                        data_input = SimpleMfccFeatures(wavsignal, fs)
                        data_label = np.array([label[genre]])
                        data_input = self.shifting(data_input)
                        data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)
                        data.append(data_input)
                        labels.append(data_label)
                return data, labels
            if mode == 'non-repetitive':
                path = self.list_path[n_start][0]
                data_label = np.array([self.list_path[n_start][1]])
                wavsignal, fs = read_wav_data(path)
                data_input = SimpleMfccFeatures(wavsignal, fs)
                data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)
                return data_input,  data_label
        else:
            print('Unknown feature type.')
            assert(0)

    def data_genetator(self, batch_size=32, epochs=0, audio_length=AUDIO_LENGTH):
        '''
        数据生成器函数，用于Keras的generator_fit训练
        batch_size: 一次产生的数据量
        '''

        assert(batch_size%CLASS_NUM==0)
        iterations_per_epoch = min(self.DataNum)//(batch_size//CLASS_NUM)+1
        self.listShuffle(min(self.DataNum))
        while True:
            ran_num = random.randint(0, iterations_per_epoch-1)  # 获取一个随机数
            origin = int(ran_num * batch_size // CLASS_NUM)
            bias = origin + epochs*min(self.DataNum)
            X,y = self.GetData(n_start=origin, n_amount=batch_size )
            X = np.array(X)
            y = np.array(y)
            yield [X, keras.utils.to_categorical(y, num_classes=self.class_num)], keras.utils.to_categorical(y, num_classes=self.class_num)  # 功能只是转成独热编码
        pass

class Testing():
    def __init__(self, feature_type, pathSame, pathDistinct):
        system_type = plat.system()  # 由于不同的系统的文件路径表示不一样，需要进行判断
        self.pathSame = pathSame[0]
        self.pathSameLabel = pathSame[1]
        self.pathDistinct = pathDistinct[0]
        self.pathDistinctLabel = pathDistinct[1]

        self.slash = ''
        if (system_type == 'Windows'):
            self.slash = '\\'  # 反斜杠
        elif (system_type == 'Linux'):
            self.slash = '/'  # 正斜杠
        else:
            print('*[Warning] Unknown System\n')
            self.slash = '/'  # 正斜杠

        if (self.slash != self.pathSame[-1]):  # 在目录路径末尾增加斜杠
            self.pathSame = self.pathSame + self.slash
        if (self.slash != self.pathDistinct[-1]):  # 在目录路径末尾增加斜杠
            self.pathDistinct = self.pathDistinct + self.slash

        self.LoadDataList()
        self.class_num = CLASS_NUM
        self.feature_type = feature_type
        self.feat_dimension = 200 if self.feature_type in ['spec','Spec','SPEC','Spectrogram','SPECTROGRAM'] else 26
        self.frame_length = 400

    def LoadDataList(self):
        self.listSame = []
        self.listDistinct = []
        link = ('bowels', 'non')
        for i in link:
            tag = 1 if i == 'bowels' else 0
            list_name_folder = os.listdir(self.pathSame + i)
            for j in list_name_folder:
                str = self.pathSame + i + self.slash + j
                self.listSame.append((str, tag))
            list_name_folder = os.listdir(self.pathDistinct + i)
            for j in list_name_folder:
                str = self.pathDistinct + i + self.slash + j
                self.listDistinct.append((str, tag))
        random.shuffle(self.listSame)
        random.shuffle(self.listDistinct)
        self.DataNum_Same = len(self.listSame)
        self.DataNum_Distinct = len(self.listDistinct)
        self.DataNum = {self.pathSameLabel:self.DataNum_Same,self.pathDistinctLabel:self.DataNum_Distinct}

    def GetData(self, n_start, n_amount=32, dataType = 'same'):
        assert(n_amount%CLASS_NUM==0)
        if dataType == self.pathSameLabel:
            path = self.listSame[n_start][0]
            data_label = np.array([self.listSame[n_start][1]])
        elif dataType == self.pathDistinctLabel:
            path = self.listDistinct[n_start][0]
            data_label = np.array([self.listDistinct[n_start][1]])
        wavsignal, fs = read_wav_data(path)
        if self.feature_type in ['spec', 'Spec', 'SPEC', 'Spectrogram', 'SPECTROGRAM']:
            data_input = GetFrequencyFeatures(wavsignal, fs, self.feat_dimension, self.frame_length, shift=160)
        elif self.feature_type in ['mfcc', 'MFCC', 'Mfcc']:
            data_input = SimpleMfccFeatures(wavsignal, fs)
        else:
            print('Unknown feature type.')
            assert (0)
        data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)
        return data_input, data_label

class Network():
    def __init__(self):
        print("Let's begin!")
        pass #it seems all the network can be described using a single function

    def CNNForest(self,feature_type, module_type, layer_counts):
        feature_length = 200 if feature_type == 'SPEC' else 26
        input_shape = (AUDIO_LENGTH, feature_length, 1)
        dictmap = {'Regular':ReguBlock, 'Residual':block, 'Inception':XcepBlock}
        Module = dictmap[module_type]
        X_input = Input(name='the_input', shape=input_shape)
        level_h1 = Module(32)(X_input)
        level_m1 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h1)  # 池化层
        level_h2 = Module(64)(level_m1)
        level_m2 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h2)  # 池化层
        level_h3 = Module(128)(level_m2)
        level_m3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h3)  # 池化层
        level_h4 = Module(256)(level_m3)
        level_m4 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h4)  # 池化层
        level_s5 = Module(512)(level_m4)
        if feature_type == 'SPEC':
            level_s5 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_s5)  # 池化层
        layers = [level_m1,level_m2,level_m3,level_m4,level_s5]
        output = layers[layer_counts//2-1]
        flayer = GlobalAveragePooling2D()(output)
        fc2 = Dense(CLASS_NUM, use_bias=True, kernel_initializer='he_normal')(flayer)  # 全连接层
        y_pred = Activation('softmax', name='Activation0')(fc2)

        model = Model(inputs=X_input, outputs=y_pred)
        optimizer = optimizers.Adadelta()
        model.compile(optimizer=optimizer, loss=[focal_loss(alpha=0.25, gamma=2)])
        print('{} cnn model with the {} feature and {} layers are estabished.'.format(module_type, feature_type, str(layer_counts)))
        modelname = feature_type+'_'+module_type+'_'+str(layer_counts)
        return model,modelname


class operation():
    def __init__(self,model,modelname,basePath):
        self.model = model
        self.clearPath = basePath
        self.basePath = os.path.join(self.clearPath,modelname)
        self.baseSavPath = []
        self.baseSavPath.append(self.basePath)
        self.baseSavPath.append(self.basePath+'_weights')

    def train(self,datapath,feature_type,batch_size=32,epoch=20):
        assert (batch_size % CLASS_NUM == 0)
        data = DataSpeech(datapath,feature_type, 'train')
        num_data = sum(data.DataNum)  # 获取数据的数�?
        os.system('pkill tensorboard')
        os.system('rm -rf ./checkpoints/files_summary/* ')
        train_writter = tf.summary.FileWriter(os.path.join(os.getcwd(), 'checkpoints', 'files_summary'))
        os.system('tensorboard --logdir=/home/zhaok14/example/PycharmProjects/setsail/individual_spp/checkpoints/files_summary/ &')
        print('\n')
        print(90 * '*')
        print(90 * '*')

        iterations_per_epoch = min(data.DataNum) // (batch_size // CLASS_NUM) + 1
        # iterations_per_epoch = 1
        print('trainer info:')
        print('training data size: %d' % num_data)
        print('increased epoches: ', epoch)
        print('minibatch size: %d' % batch_size)
        print('iterations per epoch: %d' % iterations_per_epoch)

        sess = k.get_session()
        train_writter.add_graph(sess.graph)
        sess.run(tf.global_variables_initializer())
        best_score = 0
        # epoch = 2
        duration = 0
        for i in range(0, epoch):
            iteration = 0
            yielddatas = data.data_genetator(batch_size, epoch)
            pbar = tqdm(yielddatas)
            for input, labels in pbar:
                stime = time.time()
                loss = self.model.train_on_batch(input[0], labels)
                dtime = time.time() - stime
                duration = duration + dtime
                train_summary = tf.Summary()
                train_summary.value.add(tag='loss', simple_value=loss)
                train_writter.add_summary(train_summary, iteration + i * iterations_per_epoch)
                pr = 'epoch:%d/%d,iteration: %d/%d ,loss: %s' % (epoch, i, iterations_per_epoch, iteration, loss)
                pbar.set_description(pr)
                if iteration == iterations_per_epoch:
                    break
                else:
                    iteration += 1
            pbar.close()

            self.TestModel(sess=sess, feature_type = feature_type, datapath=datapath, str_dataset='train', data_count=1000, writer=train_writter, step=i)
            metrics = self.TestModel(sess=sess, feature_type = feature_type, datapath=datapath, str_dataset='eval', data_count=-1, writer=train_writter, step=i)
            if i > 0:
                if metrics['score'] >= best_score:
                    self.metrics = metrics
                    self.metrics['epoch'] = i
                    best_score = metrics['score']
                    clrdir(self.clearPath)
                    self.savpath = []
                    self.savpath.append((self.baseSavPath[0] + '_epoch' + str(i) + '.h5'))
                    self.savpath.append((self.baseSavPath[1] + '_epoch' + str(i) + '.h5'))
                    self.model.save(self.savpath[0])
                    self.model.save_weights(self.savpath[1])
        if 'epoch' in self.metrics.keys():
            print('The best metric (without restriction) took place in the epoch: ', self.metrics['epoch'])
            print('Sensitivity: {}; Specificity: {}; Score: {}; Accuracy: {}'.format(self.metrics['sensitivity'],self.metrics['specificity'],self.metrics['score'],self.metrics['accuracy']))
            self.TestGenerability(feature_type = feature_type, weightspath=self.savpath[1])
        else:
            print('The restricted best metric is not found. Done!')
        print('Training duration: {}s'.format(round(duration, 2)))

    def TestModel(self, sess, writer, feature_type, datapath='', str_dataset='eval', data_count=32, show_ratio=True, step=0):
        '''
        测试检验模型效果
        '''
        data = DataSpeech(datapath, feature_type, str_dataset)
        num_data = sum(data.DataNum)  # 获取数据的数量
        if (data_count <= 0 or data_count > num_data):  # 当data_count为小于等于0或者大于测试数据量的值时，则使用全部数据来测试
            data_count = num_data
        try:
            ran_num = random.randint(0, num_data - 1)  # 获取一个随机数
            overall_p = 0
            overall_n = 0
            overall_tp = 0
            overall_tn = 0
            start = time.time()
            cm_pre = []
            cm_lab = []
            map = {0: 'normal', 1: 'bowel sounds'}
            # data_count = 200
            for i in tqdm(range(data_count)):
                data_input, data_labels = data.GetData((ran_num + i) % num_data, mode='non-repetitive')  # 从随机数开始连续向后取一定数量数据
                data_pre = self.model.predict_on_batch(np.expand_dims(data_input, axis=0))
                predictions = np.argmax(data_pre[0], axis=0)
                cm_pre.append(map[predictions])
                cm_lab.append(map[data_labels[0]])
                tp, fp, tn, fn = Compare2(predictions, data_labels[0])  # 计算metrics
                overall_p += tp + fn
                overall_n += tn + fp
                overall_tp += tp
                overall_tn += tn
            if overall_p != 0:
                sensitivity = overall_tp / overall_p * 100
                sensitivity = round(sensitivity, 2)
            else:
                sensitivity = 'None'
            if overall_n != 0:
                specificity = overall_tn / overall_n * 100
                specificity = round(specificity, 2)
            else:
                specificity = 'None'
            if sensitivity != 'None' and specificity != 'None':
                score = (sensitivity + specificity) / 2
                score = round(score, 2)
            else:
                score = 'None'
            accuracy = (overall_tp + overall_tn) / (overall_p + overall_n) * 100
            accuracy = round(accuracy, 2)
            end = time.time()
            dtime = round(end - start, 2)
            strg = '*[测试结果] 片段识别 {0} 敏感度：{1}%, 特异度： {2}%, 得分： {3}, 准确度： {4}%, 用时: {5}s.'.format(str_dataset,sensitivity,specificity, score,accuracy, dtime)
            tqdm.write(strg)

            assert (len(cm_lab) == len(cm_pre))
            img_cm = plot_confusion_matrix(cm_lab, cm_pre, list(map.values()),tensor_name='MyFigure/cm', normalize=False)
            writer.add_summary(img_cm, global_step=step)
            summary = tf.Summary()
            summary.value.add(tag=str_dataset + '/sensitivity', simple_value=sensitivity)
            summary.value.add(tag=str_dataset + '/specificity', simple_value=specificity)
            summary.value.add(tag=str_dataset + '/score', simple_value=score)
            summary.value.add(tag=str_dataset + '/accuracy', simple_value=accuracy)
            writer.add_summary(summary, global_step=step)

            metrics = {'data_set': str_dataset, 'sensitivity': sensitivity, 'specificity': specificity, 'score': score,'accuracy': accuracy}
            return metrics

        except StopIteration:
            print('*[Error] Model Test Error. please check data format.')

    def TestGenerability(self, feature_type, weightspath, datasourceA=None, datasourceB=None):
        training = ['/home/zhaok14/example/PycharmProjects/setsail/individual_spp/bowelsounds/perfect/train/','train']
        validation = ['/home/zhaok14/example/PycharmProjects/setsail/individual_spp/bowelsounds/perfect/validation/','validation']
        test_same = ['/home/zhaok14/example/PycharmProjects/setsail/individual_spp/bowelsounds/perfect/test-same','same']
        test_different = ['/home/zhaok14/example/PycharmProjects/setsail/individual_spp/bowelsounds/perfect/test-0419-different/','different']

        print(90 * '*')
        print('Firstly self-check the generability testing method:')
        self.__dataTesting__(feature_type, training, validation, weightspath)
        print('')
        print('Then derive the generability testing results: ')
        self.__dataTesting__(feature_type, test_same, test_different, weightspath)

    def __dataTesting__(self, feature_type, dataSourceA, dataSourceB, weightspath):
        data = Testing(feature_type, dataSourceA, dataSourceB)
        self.model.load_weights(weightspath)
        for choice in (dataSourceA[1], dataSourceB[1]):
            num_data = data.DataNum[choice]  # 获取数据的数量
            ran_num = random.randint(0, num_data - 1)  # 获取一个随机数
            overall_p = 0
            overall_n = 0
            overall_tp = 0
            overall_tn = 0

            start = time.time()
            # data_count = 200
            pbar = tqdm(range(num_data))
            for i in pbar:
                data_input, data_labels = data.GetData((ran_num + i) % num_data, dataType=choice)  # 从随机数开始连续向后取一定数量数据
                data_pre = self.model.predict_on_batch(np.expand_dims(data_input, axis=0))
                predictions = np.argmax(data_pre[0], axis=0)
                tp, fp, tn, fn = Compare2(predictions, data_labels[0])  # 计算metrics
                overall_p += tp + fn
                overall_n += tn + fp
                overall_tp += tp
                overall_tn += tn

            if overall_p != 0:
                sensitivity = overall_tp / overall_p * 100
                sensitivity = round(sensitivity, 2)
            else:
                sensitivity = 'None'
            if overall_n != 0:
                specificity = overall_tn / overall_n * 100
                specificity = round(specificity, 2)
            else:
                specificity = 'None'
            if sensitivity != 'None' and specificity != 'None':
                score = (sensitivity + specificity) / 2
                score = round(score, 2)
            else:
                score = 'None'
            accuracy = (overall_tp + overall_tn) / (overall_p + overall_n) * 100
            accuracy = round(accuracy, 2)
            end = time.time()
            dtime = round(end - start, 2)
            strg = '*[泛化性测试结果] 片段类型【{0}】 敏感度：{1}%, 特异度： {2}%, 得分： {3}, 准确度： {4}%, 用时: {5}s.'.format(choice, sensitivity, specificity, score, accuracy, dtime)
            tqdm.write(strg)
            pbar.close()
import sys
class logger(object):
    def __init__(self,filename=os.path.join(os.getcwd(),'Default.log')):
        self.terminal = sys.stdout
        self.log = open(filename,'w')
    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

from keras.backend.tensorflow_backend import set_session
import gc
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #only display error and warning; for 1: all info; for 3: only error.
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显�? 按需分配
set_session(tf.Session(config=config))
datapath = '/home/zhaok14/example/PycharmProjects/setsail/individual_spp/bowelsounds/perfect'
SUMMARY = True
if SUMMARY:
    nn = Network()
    model, name = nn.CNNForest('MFCC', 'Inception', 6)
    model.summary()
else:
    sys.stdout = logger()
    folders = pathpoper()
    nn = Network()
    st = time.time()
    for feature_type in ('spec','mfcc'):
        for module_type in ('regular','residual','inception'):
            for layer_counts in (2,4,6,8,10):
                print()
                print(40 * '-'+'HANDLEFLAG'+40 * '-')
                print()
                feature, module, layer = stringCheck(feature_type,module_type,layer_counts)
                path = folders.popup(feature,module,layer)
                model,name = nn.CNNForest(feature,module,layer)
                controller = operation(model,name,path)
                controller.train(datapath,feature)
                gc.collect()
    en = time.time()-st
    hour = en // 3600
    minute = ( en - (hour * 3600) ) //60
    seconds = en - (hour * 3600) - (minute * 60)
    print('Overall design time: {}s, i.e., {} hour(s), {} minute(s), {}s'.format(en,int(hour),int(minute),seconds))