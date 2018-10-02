#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import platform as plat
import re  # regex
from tqdm import tqdm
import itertools
import tfplot
import matplotlib
from sklearn.metrics import confusion_matrix
from textwrap import wrap
import gc

from general_func.file_wav import *
from general_func.gen_func import Comapare2

# LSTM_CNN
import keras as kr
import numpy as np
import random
import tensorflow as tf

from readdata_5folds import DataCross
from readdata_5folds import MAX_AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, CLASS_NUM, FOLDER_SPLIT_NUM
from collections import Counter


class ModelSpeech():  # 语音模型类
    def __init__(self, datapath):
        '''
        初始化
        默认输出四类：normal,wheezes,crackles,both
        '''
        MS_OUTPUT_SIZE = 4
        self.MS_OUTPUT_SIZE = MS_OUTPUT_SIZE  # 神经网络最终输出的每一个字符向量维度的大小
        self.label_max_string_length = 64
        self.AUDIO_LENGTH = MAX_AUDIO_LENGTH
        self.AUDIO_FEATURE_LENGTH = AUDIO_FEATURE_LENGTH

        self.train_summary=self.CreateModel()

        self.datapath = datapath
        self.slash = ''
        system_type = plat.system()  # 由于不同的系统的文件路径表示不一样，需要进行判断
        if (system_type == 'Windows'):
            self.slash = '\\'  # 反斜杠
        elif (system_type == 'Linux'):
            self.slash = '/'  # 正斜杠
        else:
            print('*[Message] Unknown System\n')
            self.slash = '/'  # 正斜杠
        if (self.slash != self.datapath[-1]):  # 在目录路径末尾增加斜杠
            self.datapath = self.datapath + self.slash

        self.metrics = 0

    def CreateModel(self):

        self.input_data = tf.placeholder(dtype=tf.float32, shape=[None, self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH, 1])
        self.label = tf.placeholder(dtype=tf.float32, shape=[None, 4])
        self.is_train = tf.placeholder(dtype=tf.bool)

        conv2d_1 = tf.layers.conv2d(self.input_data, 32, (3, 3), activation=tf.nn.relu,  use_bias=True, padding='same',
                                    kernel_initializer=tf.keras.initializers.he_normal(), name='conv2d_1')
        droput_1 = tf.layers.dropout(conv2d_1, rate=0.1, training=self.is_train)
        conv2d_2 = tf.layers.conv2d(droput_1, 32, (3, 3), activation=tf.nn.relu, use_bias=True, padding='same',
                                    kernel_initializer=tf.keras.initializers.he_normal(), name='conv2d_2')
        max_pool_2 = tf.layers.max_pooling2d(conv2d_2, pool_size=2, strides=2, padding="valid", name='max_pool_2')
        droput_2 = tf.layers.dropout(max_pool_2, rate=0.1, training=self.is_train)

        conv2d_3 = tf.layers.conv2d(droput_2, 64, (3, 3), activation=tf.nn.relu, use_bias=True, padding='same',
                                    kernel_initializer=tf.keras.initializers.he_normal(), name='conv2d_3')  # 卷积层
        droput_3 = tf.layers.dropout(conv2d_3, rate=0.2, training=self.is_train)  # 随机丢失层
        conv2d_4 = tf.layers.conv2d(droput_3, 64, (3, 3), activation=tf.nn.relu, use_bias=True, padding='same',
                                    kernel_initializer=tf.keras.initializers.he_normal(), name='conv2d_4')  # 卷积层
        max_pool_4 = tf.layers.max_pooling2d(conv2d_4, pool_size=2, strides=2, padding="valid", name='max_pool_4')
        droput_4 = tf.layers.dropout(max_pool_4, rate=0.2, training=self.is_train)

        conv2d_5 = tf.layers.conv2d(droput_4, 128, (3, 3), activation=tf.nn.relu, use_bias=True, padding='same',
                                    kernel_initializer=tf.keras.initializers.he_normal(), name='conv2d_5')
        droput_5 = tf.layers.dropout(conv2d_5, rate=0.3, training=self.is_train)
        conv2d_6 = tf.layers.conv2d(droput_5, 128, (3, 3), activation=tf.nn.relu, use_bias=True, padding='same',
                                    kernel_initializer=tf.keras.initializers.he_normal(), name='conv2d_6')
        max_pool_6 = tf.layers.max_pooling2d(conv2d_6, pool_size=2, strides=2, padding="valid", name='max_pool_6')
        droput_6 = tf.layers.dropout(max_pool_6, rate=0.4, training=self.is_train)

        conv2d_7 = tf.layers.conv2d(droput_6, 128, (3, 3), activation=tf.nn.relu, use_bias=True, padding='same',
                                    kernel_initializer=tf.keras.initializers.he_normal(), name='conv2d_7')
        droput_7 = tf.layers.dropout(conv2d_7, rate=0.4, training=self.is_train)
        conv2d_8 = tf.layers.conv2d(droput_7, 128, (3, 3), activation=tf.nn.relu, use_bias=True, padding='same',
                                    kernel_initializer=tf.keras.initializers.he_normal(), name='conv2d_8')

        self.flat_layer = tf.layers.flatten(conv2d_8)

        droput_9 = tf.layers.dropout(self.flat_layer, rate=0.4, training=self.is_train)

        fc_1 = tf.layers.dense(droput_9, 128, activation=tf.keras.activations.relu, use_bias=True,
                                              kernel_initializer=tf.keras.initializers.he_normal(),
                                              name='fc_1')

        droput_10 = tf.layers.dropout(fc_1, rate=0.5, training=self.is_train)
        self.fc_2 = tf.layers.dense(droput_10, self.MS_OUTPUT_SIZE, activation=None, use_bias=True,
                               kernel_initializer=tf.keras.initializers.he_normal(),
                               name='fc_2')

        self.y_pred = tf.keras.activations.softmax(self.fc_2)
        # self.y_pred = tf.clip_by_value(y_pred, 1e-10, 1.0)
        # self.loss = - tf.reduce_sum( self.label * tf.log(self.y_pred) )
        self.loss = tf.losses.softmax_cross_entropy(self.label,self.fc_2)
        tv1=tf.summary.scalar('loss', self.loss)

        self.optimize = tf.train.AdadeltaOptimizer(learning_rate=0.01, rho=0.95, epsilon=1e-06).minimize(self.loss)

        with tf.variable_scope('fc_1', reuse=True):
            w = tf.get_variable('kernel')
            tv2=tf.summary.histogram('fc1',w)

        return [tv1,tv2]

    def TrainModel(self, datapath, epoch=2, batch_size=32, load_model=False, filename='model_set/speech_model25'):
        '''
        训练模型
        参数：
            datapath: 数据保存的路径
            epoch: 迭代轮数
            save_step: 每多少步保存一次模型
            filename: 默认保存文件名，不含文件后缀名
        '''
        '''
        currently dont involve in load model function and txt export function.
        '''
        assert(batch_size%4 == 0)
        data = DataCross(datapath)
        #1. clear the memory
        #2. initialize the paras
        #3. training, testing
        #4. recording
        #5. after all, statistic the info
        for fold in range(FOLDER_SPLIT_NUM):
            print('the {}th cross validation'.format(fold))
            gc.collect()
            data.SplitType(type='train',order=fold)
            num_data = sum(data.DataNum)  # 获取数据的数量
            # iterations_per_epoch = 2
            iterations_per_epoch = max(data.DataNum) // (batch_size//4) + 1
            print('trainer info:')
            print('training data amounts: %d' % num_data)
            print('increased epoches: ', epoch)
            print('minibatch size: %d' % batch_size)
            print('iterations per epoch: %d' % iterations_per_epoch)

            saver = tf.train.Saver(max_to_keep=2)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                if load_model == True:
                    try:
                        saver.restore(sess, os.path.join(os.getcwd(), 'checkpoints', 'files_model', 'speech.module')) #two files in a folder.
                    except:
                        print('Loading weights failed. Train from scratch.')
                summary_merge = tf.summary.merge(self.train_summary)
                os.system('rm -rf ./checkpoints/files_summary ')
                train_writter = tf.summary.FileWriter(os.path.join(os.getcwd(), 'checkpoints', 'files_summary'), sess.graph)
                best_score = 0
                for i in range(0, epoch):
                    iteration = 0
                    yielddatas = data.data_genetator4tTrain(batch_size)
                    pbar = tqdm(yielddatas)
                    for input, _ in pbar:
                        feed = {self.input_data: input[0], self.label: input[1], self.is_train: True}
                        _, loss, train_summary = sess.run([self.optimize, self.loss, summary_merge], feed_dict=feed)
                        train_writter.add_summary(train_summary, iteration + i * iterations_per_epoch)
                        pr = 'epoch:%d/%d,iteration: %d/%d ,loss: %s' % (epoch, i, iterations_per_epoch, iteration, loss)
                        pbar.set_description(pr)
                        if iteration == iterations_per_epoch:
                            break
                        else:
                            iteration += 1
                    pbar.close()
                    self.TestModel(sess=sess, datapath=data, type='train', data_count=-1, out_report=False, writer=train_writter, step=i)
                    metrics = self.TestModel(sess=sess, datapath=self.datapath, str_dataset='eval', data_count=-1, out_report=False, writer=train_writter, step=i)
                    if (metrics['score'] > best_score and i>19):
                        self.metrics = metrics
                        self.metrics['epoch'] = i
                        best_score =metrics['score']
                        saver.save(sess, os.path.join(os.getcwd(),'checkpoints','files_model', 'speech.module'), global_step=i)

        print('The best metrics took place in the epoch: ', self.metrics['epoch'])
        print('Sensitivity: {}; Specificity: {}; Score: {}; Accuracy: {}'.format(self.metrics['sensitivity'],self.metrics['specificity'],self.metrics['score'],self.metrics['accuracy']))

    def SaveModel(self, filename='model_set/speech_model25', comment=''):
        '''
        保存模型参数
        '''
        self.model.save_weights(filename + comment + '.model')
        f = open('step25.txt', 'w')
        f.write(filename + comment)
        f.close()

    def TestModel(self, sess, dataObject='', type='eval', data_count=32, out_report=False, show_ratio=True,writer=tf.summary.FileWriter('files_summary', tf.get_default_graph()),step=0):
        '''
        测试检验模型效果
        '''
        data = DataSpeech(self.datapath, str_dataset)
        # data.LoadDataList(str_dataset)
        num_data = sum(data.DataNum)  # 获取数据的数量
        if (data_count <= 0 or data_count > num_data):  # 当data_count为小于等于0或者大于测试数据量的值时，则使用全部数据来测试
            data_count = num_data

        try:
            ran_num = random.randint(0, num_data - 1)  # 获取一个随机数
            # ran_num = num_data // 2 + 1
            # print('\n\n It is fixed test now! spot:%d' % ran_num)

            overall_p = 0
            overall_n = 0
            overall_tp = 0
            overall_tn = 0
            accuracy = 0
            sensitivity = 0
            specificity = 0
            score = 0


            nowtime = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
            txt_obj = []
            if (out_report == True):
                txt_obj = open('Test_Report_' + str_dataset + '_' + nowtime + '.txt', 'w', encoding='UTF-8')  # 打开文件并读入

            start = time.time()
            cm_pre = []
            cm_lab = []
            map = {0:'normal',1:'wheeze',2:'crackle',3:'both'}
            for i in range(data_count):
                if (i % 100 == 0 and show_ratio == True):
                    strg = '测试进度：{0}/{1}'.format(i,data_count)
                    # print('测试进度：', i, '/', data_count)
                    tqdm.write(strg)

                data_input, data_labels = data.GetData((ran_num + i) % num_data,
                                                       mode='non-repetitive')  # 从随机数开始连续向后取一定数量数据

                predictions = []
                if len(data_input) <= self.AUDIO_LENGTH:
                    data_in = np.zeros((1,self.AUDIO_LENGTH,self.AUDIO_FEATURE_LENGTH,1), dtype=np.float)
                    data_in[0,0:len(data_input)] = data_input
                    feed = {self.input_data: data_in, self.label: kr.utils.to_categorical(data_labels,num_classes=CLASS_NUM),
                        self.is_train: False}
                    data_pre = sess.run([self.y_pred], feed_dict=feed)
                    predictions = np.argmax(data_pre[0][0],axis=0)
                else:
                    data_in = np.zeros((5, self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH, 1), dtype=np.float)
                    span = len(data_input) - self.AUDIO_LENGTH
                    for t in range(5):
                        begin = np.random.randint(span)
                        data_input = data_input[begin:begin + self.AUDIO_LENGTH]
                        data_in[t, 0:len(data_input)] = data_input
                    feed = {self.input_data: data_in,
                            self.label: kr.utils.to_categorical(data_labels, num_classes=CLASS_NUM),
                            self.is_train: False}
                    data_pre = sess.run([self.y_pred], feed_dict=feed)
                    tmp = np.argmax(data_pre[0], axis=1)
                    predictions=Counter(tmp).most_common(1)[0][0]

                # print('predictions:',predictions)
                # print('data_pre:',np.argmax(data_pre[0][0],axis=0))
                # print ('data_label:',data_labels[0])

                cm_pre.append(map[predictions])
                cm_lab.append(map[data_labels[0]])

                tp, fp, tn, fn = Comapare2(predictions, data_labels[0])  # 计算metrics
                overall_p += tp + fn
                overall_n += tn + fp
                overall_tp += tp
                overall_tn += tn

                txt = ''
                if (out_report == True):
                    txt += str(i) + '\n'
                    txt += 'True:\t' + str(data_labels) + '\n'
                    txt += 'Pred:\t' + str(data_pre) + '\n'
                    txt += '\n'
                    txt_obj.write(txt)

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
            accuracy = (overall_tp+overall_tn) / (overall_p+overall_n) * 100
            accuracy = round(accuracy, 2)
            end = time.time()
            dtime = round(end-start,2)
            # print('*[测试结果] 片段识别 ' + str_dataset + ' 敏感度：', sensitivity, '%, 特异度： ', specificity, '%, 得分： ', score, ', 准确度： ', accuracy, '%, 用时: ', dtime, 's.')
            strg = '*[测试结果] 片段识别 {0} 敏感度：{1}%, 特异度： {2}%, 得分： {3}, 准确度： {4}%, 用时: {5}s.'.format(str_dataset, sensitivity, specificity, score, accuracy, dtime)
            tqdm.write(strg)

            assert(len(cm_lab) == len(cm_pre))
            img_cm = self.plot_confusion_matrix(cm_lab, cm_pre, list(map.values()),
                                  tensor_name='MyFigure/cm', normalize=False)
            writer.add_summary(img_cm,global_step=step)
            summary = tf.Summary()
            summary.value.add(tag=str_dataset + '/sensitivity',simple_value=sensitivity )
            summary.value.add(tag=str_dataset + '/specificity',simple_value=specificity )
            summary.value.add(tag=str_dataset + '/score', simple_value=score )
            summary.value.add(tag=str_dataset + '/accuracy', simple_value=accuracy)
            writer.add_summary(summary,global_step=step)

            if (out_report == True):
                txt = '*[测试结果] 片段识别 ' + str_dataset + ' 敏感度：' + sensitivity + '%, 特异度： ' + specificity + '%, 得分： ' + score + ', 准确度： ' + accuracy + '%, 用时: ' + dtime + 's.'
                txt_obj.write(txt)
                txt_obj.close()

            metrics = {'data_set':str_dataset,'sensitivity':sensitivity,'specificity':specificity,'score':score,'accuracy':accuracy}

            return metrics

        except StopIteration:
            print('[Error] Model Test Error. please check data format.')

    def plot_confusion_matrix(self, correct_labels, predict_labels, labels, title='Confusion matrix',
                              tensor_name='MyFigure/image', normalize=False):
        '''
        Parameters:
            correct_labels                  : These are your true classification categories.
            predict_labels                  : These are you predicted classification categories
            labels                          : This is a lit of labels which will be used to display the axix labels
            title='Confusion matrix'        : Title for your matrix
            tensor_name = 'MyFigure/image'  : Name for the output summay tensor

        Returns:
            summary: TensorFlow summary

        Other itema to note:
            - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc.
            - Currently, some of the ticks dont line up due to rotations.
        '''
        cm = confusion_matrix(correct_labels, predict_labels, labels=labels)
        if normalize:
            cm = cm.astype('float') * 10 / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm, copy=True)
            cm = cm.astype('int')

        np.set_printoptions(precision=2)

        fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(cm, cmap='Oranges')

        classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
        classes = ['\n'.join(wrap(l, 40)) for l in classes]

        tick_marks = np.arange(len(classes))

        ax.set_xlabel('Predictions ', fontsize=25)
        ax.set_xticks(tick_marks)
        c = ax.set_xticklabels(classes, fontsize=20, rotation=-90, ha='center')
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()

        ax.set_ylabel('Truths', fontsize=25)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, fontsize=20, va='center')
        ax.yaxis.set_label_position('left')
        ax.yaxis.tick_left()

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], 'd') if cm[i, j] != 0 else '.', horizontalalignment="center", fontsize=25,
                    verticalalignment='center', color="black")
        fig.set_tight_layout(True)
        summary = tfplot.figure.to_summary(fig, tag=tensor_name)  # Convert a matplotlib figure fig into a TensorFlow Summary object that can be directly fed into Summary.FileWriter
        return summary

    def Predict(self, data_input):
        '''
        预测结果
        返回语音识别后的拼音符号列表
        '''
        x_in = data_input.reshape(1, data_input.shape[0], data_input.shape[1], data_input.shape[2])

        base_pred = self.model_data.predict(x=x_in)
        pred_class = np.argmax(base_pred, axis=1)
        pred_class = pred_class.tolist()
        return pred_class

        pass


if (__name__ == '__main__'):

    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # 进行配置，使用70%的GPU
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.93
    # config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
    set_session(tf.Session(config=config))

    datapath = ''
    modelpath = 'model_set'

    if (not os.path.exists(modelpath)):  # 判断保存模型的目录是否存在
        os.makedirs(modelpath)  # 如果不存在，就新建一个，避免之后保存模型的时候炸掉

    system_type = plat.system()  # 由于不同的系统的文件路径表示不一样，需要进行判断
    if (system_type == 'Windows'):
        datapath = 'E:\\语音数据集'
        modelpath = modelpath + '\\'
    elif (system_type == 'Linux'):
        datapath = 'dataset'
        modelpath = modelpath + '/'
    else:
        print('*[Message] Unknown System\n')
        datapath = 'dataset'
        modelpath = modelpath + '/'

    ms = ModelSpeech(datapath)

