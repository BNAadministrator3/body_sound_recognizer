#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import platform as plat

from general_func.file_wav import *
from general_func.gen_func import Comapare

# LSTM_CNN
import numpy as np
import random
import tensorflow as tf

from keras.models import Model
from keras.layers import Dense, Dropout, Input  # , Flatten,LSTM,Convolution1D,MaxPooling1D
from keras.layers import Lambda, Conv2D, MaxPooling2D  #, Merge,Conv1D
from keras import backend as K
from keras.optimizers import Adadelta
from spp.SpatialPyramidPooling import SpatialPyramidPooling

from model_set.readdata import DataSpeech

from model_set.readdata import MAX_AUDIO_LENGTH,AUDIO_FEATURE_LENGTH

def as_keras_metric(method):
    import functools
    from keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

precision = as_keras_metric(tf.metrics.precision)
recall = as_keras_metric(tf.metrics.recall)
false_negatives =as_keras_metric(tf.metrics.false_negatives)
false_positives =as_keras_metric(tf.metrics.false_positives)
true_positives =as_keras_metric(tf.metrics.true_positives)
true_negatives =as_keras_metric(tf.metrics.true_negatives)


class ModelSpeech(): # 语音模型类
    def __init__(self, datapath):
        '''
        初始化
        默认输出四类：normal,wheezes,crackles,both
        '''
        MS_OUTPUT_SIZE = 4
        self.MS_OUTPUT_SIZE = MS_OUTPUT_SIZE # 神经网络最终输出的每一个字符向量维度的大小
        self.label_max_string_length = 64
        self.AUDIO_LENGTH = MAX_AUDIO_LENGTH
        self.AUDIO_FEATURE_LENGTH = AUDIO_FEATURE_LENGTH
        self.model = self.CreateModel()

        self.datapath = datapath
        self.slash = ''
        system_type = plat.system() # 由于不同的系统的文件路径表示不一样，需要进行判断
        if(system_type == 'Windows'):
            self.slash='\\' # 反斜杠
        elif(system_type == 'Linux'):
            self.slash='/' # 正斜杠
        else:
            print('*[Message] Unknown System\n')
            self.slash='/' # 正斜杠
        if(self.slash != self.datapath[-1]): # 在目录路径末尾增加斜杠
            self.datapath = self.datapath + self.slash

    def CreateModel(self):
        '''
        定义CNN/LSTM/CTC模型，使用函数式模型 #note is concerning the mfcc like process.
        输入层：39维的特征值序列，一条语音数据的最大长度设为1500（大约15s）
        隐藏层一：1024个神经元的卷积层
        隐藏层二：池化层，池化窗口大小为2
        隐藏层三：Dropout层，需要断开的神经元的比例为0.2，防止过拟合
        隐藏层四：循环层、LSTM层
        隐藏层五：Dropout层，需要断开的神经元的比例为0.2，防止过拟合
        隐藏层六：全连接层，神经元数量为self.MS_OUTPUT_SIZE，使用softmax作为激活函数，
        输出层：自定义层，即CTC层，使用CTC的loss作为损失函数，实现连接性时序多输出

        '''
        self.input_data = Input(name='the_input', shape=(None, self.AUDIO_FEATURE_LENGTH, 1))
        self.labels = Input(name='the_labels', shape=(4,))
        
        layer_h1 = Conv2D(32, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(self.input_data) # 卷积层
        layer_h1 = Dropout(0.1)(layer_h1)
        layer_h2 = Conv2D(32, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h1) # 卷积层
        layer_h3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h2) # 池化层
        layer_h3 = Dropout(0.1)(layer_h3)
        
        layer_h4 = Conv2D(64, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h3) # 卷积层
        layer_h4 = Dropout(0.2)(layer_h4)
        layer_h5 = Conv2D(64, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h4) # 卷积层
        layer_h6 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h5) # 池化层
        layer_h6 = Dropout(0.2)(layer_h6)
        
        layer_h7 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h6) # 卷积层
        layer_h7 = Dropout(0.3)(layer_h7)
        layer_h8 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h7) # 卷积层
        layer_h9 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h8) # 池化层
        layer_h9 = Dropout(0.3)(layer_h9)
      
        layer_h10 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h9) # 卷积层
        layer_h10 = Dropout(0.4)(layer_h10)
        layer_h11 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h10) # 卷积层
        self.layer_h12 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h11) # 池化层

        #用spp就可以变成一张特征图，且size固定
        self.layer_s12 = SpatialPyramidPooling([1,2,4])(self.layer_h12)
    
        layer_d13 = Dropout(0.4)(self.layer_s12)
        self.layer_h14 = Dense(128, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_d13) # 全连接层
        layer_d14 = Dropout(0.5)(self.layer_h14)
        self.y_pred = Dense(self.MS_OUTPUT_SIZE, use_bias=True, kernel_initializer='he_normal',activation='softmax')(layer_d14) # 全连接层
        self.model_data = Model(inputs=self.input_data, outputs=self.y_pred)
        
        loss_out = Lambda(self.softmax_lambda_func, output_shape=(1,), name='softmax_loss')(
            [self.y_pred, self.labels])
        model = Model(inputs=[self.input_data, self.labels], outputs=loss_out)
        ada_d = Adadelta(lr = 0.01, rho = 0.95, epsilon = 1e-06)
        model.compile(loss={'softmax_loss': lambda y_true, y_pred: y_pred}, optimizer = ada_d, metrics=[false_positives,false_negatives,true_positives,true_negatives])

        print('[*提示] 创建模型成功，模型编译成功')
        return model
   
    def softmax_lambda_func(self, args):
        y_pred, labels = args
        loss = -labels*K.log( tf.clip_by_value(y_pred,1e-10,1.0)  )
        # loss = -labels*(1-tf.clip_by_value(y_pred,1e-10,1.0)) * (1-tf.clip_by_value(y_pred,1e-10,1.0)) * K.log( tf.clip_by_value(y_pred,1e-10,1.0) )

        return loss
   
   
    def TrainModel(self, datapath, epoch = 2, batch_size = 32, load_model = False, filename = 'model_set/speech_model25'):
        '''
        训练模型
        参数：
            datapath: 数据保存的路径
            epoch: 迭代轮数
            save_step: 每多少步保存一次模型
            filename: 默认保存文件名，不含文件后缀名
        '''
        self.TestModel(self.datapath, str_dataset='train', data_count=4, out_report=False)
        prepoch = 0
        if load_model:
            prepoch = self.LoadModel(filename=filename)

        data=DataSpeech(datapath, 'train')
        num_data = sum(data.DataNum)  # 获取数据的数量
        yielddatas = data.data_genetator(batch_size)
        iterations_per_epoch = num_data // batch_size + 1
       
        for epoches in range(epoch):  # 迭代轮数
            print('trainer info:')
            print('training data size: %d' % num_data)
            print('increased epoches: ',epoch)
            print('minibatch size: %d' % batch_size)
            print('iterations per epoch: %d' % iterations_per_epoch)
            print('[running] training epoch: %d ..' % (epoches+1+prepoch))

            try:
                print('[message] epoch %d . Have trained %d iterations' % (
                epoches+1+prepoch, max(0, (epoches+prepoch) * iterations_per_epoch)))
                # data_genetator是一个生成器函数
                self.model.fit_generator(yielddatas, steps_per_epoch=iterations_per_epoch)
                # sess = K.get_session()
                # f = open('log_record.txt', 'a')
                # for a,_ in yielddatas:
                #     print('训练')
                #     loss_out = sess.run(self.layer_r12,feed_dict={self.input_data:a[0],self.input_rois:a[1],self.labels:a[2]})
                #     loss_out2= sess.run(self.layer_m12,feed_dict={self.input_data:a[0],self.input_rois:a[1],self.labels:a[2]})
                #     loss_out3 = sess.run(self.layer_h13,feed_dict={self.input_data:a[0],self.input_rois:a[1],self.labels:a[2]})
                #     txt = '分隔符：============\n'
                    # txt += 'loss_out\n'
                    # txt += str(loss_out)
                    # txt += '\ny_pred\n'
                    # txt += str(y_pred)
                    # txt += '\nloss\n'
                    # txt += str(loss)
                
            except StopIteration:
                print('[error] Generator error. Please check data formats.')
                break
            # one epoch to save and validate
            os.system('rm model_set/* ')
            self.SaveModel(comment='_new_' + str(epoches+prepoch+1) + '_steps_' + str((epoches+prepoch+1) * iterations_per_epoch))
            self.TestModel(self.datapath, str_dataset='train', data_count=-1, out_report = False)

    def LoadModel(self,auto=True,filename='model_set/speech_model25.model'):
        '''
        加载模型参数
        '''
        try:
            prepoch = 0
            if auto:
                strs = os.listdir('model_set')
                if len(strs)!=1:
                    print('[warning] Loading weights fails. Train the model from scratch..')
                    return int(0)
                else:
                    self.model.load_weights('model_set' + self.slash + strs[0])
                    #返回preepoch
                    sindex = strs[0].find('new_')
                    eindex = strs[0].find('_steps')
                    prepoch = int(strs[0][sindex+4:eindex])
                    print('[message] Weights loaded automatically. ')
            else:
                #先不管
                self._model.load_weights(filename) #sdsd
                self.base_model.load_weights(filename + '.base') #有两个后缀名
                sindex = filename.find('new_')
                eindex = filename.find('_steps')
                prepoch = int(filename[sindex + 4:eindex])
                print('[message] Weights loaded. ')
            print('[message] so far already trained epoches:', prepoch)
            return prepoch
        except:
            prepoch = int(0)
            print('[warning] Loading fails presumbly due to model alteration. ')
            return prepoch

    def SaveModel(self,filename='model_set/speech_model25',comment=''):
        '''
        保存模型参数
        '''
        self.model.save_weights(filename+comment+'.model')
        f = open('step25.txt','w')
        f.write(filename+comment)
        f.close()

    def TestModel(self, datapath='', str_dataset='eval', data_count = 32, out_report = False, show_ratio = True):
        '''
        测试检验模型效果
        '''
        data=DataSpeech(self.datapath, str_dataset)
        #data.LoadDataList(str_dataset)
        num_data = sum(data.DataNum) # 获取数据的数量
        if(data_count <= 0 or data_count > num_data): # 当data_count为小于等于0或者大于测试数据量的值时，则使用全部数据来测试
            data_count = num_data

        try:
            ran_num = random.randint(0,num_data - 1) # 获取一个随机数
            ran_num = num_data//2 + 1
            print('It is fixed test now! spot:%d'%ran_num)

            overall_p = 0
            overall_n = 0
            overall_tp = 0
            overall_tn = 0
            sensitivity = 0
            specificity = 0

            nowtime = time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))
            txt_obj = []
            if(out_report == True):
                txt_obj = open('Test_Report_' + str_dataset + '_' + nowtime + '.txt', 'w', encoding='UTF-8') # 打开文件并读入

            for i in range(data_count):
                data_input, data_labels = data.GetData((ran_num + i) % num_data, mode='non-repetitive')  # 从随机数开始连续向后取一定数量数据
                num_bias = 0
                while(data_input.shape[0] > self.AUDIO_LENGTH):
                    print('*[Error]','.wav data with NO. ',(ran_num + i) % num_data, 'is too long with length ', data_input.shape[0], ',\n A Exception raises when test Speech Model.')
                    num_bias += 1
                    data_input, data_labels = data.GetData((ran_num + i + num_bias) % num_data, mode='non-repetitive')  # 从随机数开始连续向后取一定数量数据

                if (i % 100 == 0 and show_ratio == True):
                    print('测试进度：', i, '/', data_count)

                data_pre = self.Predict(data_input)
                # print('data_pre:',data_pre)
                # print ('data_label:',data_labels.tolist())

                tp,fp,tn,fn = Comapare(data_pre,data_labels) # 计算metrics
                overall_p += tp+fn
                overall_n += tn+fp
                overall_tp += tp
                overall_tn += tn

                txt = ''
                if(out_report == True):
                    txt += str(i) + '\n'
                    txt += 'True:\t' + str(data_labels) + '\n'
                    txt += 'Pred:\t' + str(data_pre) + '\n'
                    txt += '\n'
                    txt_obj.write(txt)
             
            if overall_p != 0:
                sensitivity = overall_tp / overall_p * 100
                sensitivity = round(sensitivity,2)
            else:
                sensitivity = 'None'
            if overall_n != 0:
                specificity = overall_tn / overall_n * 100
                specificity = round(specificity,2)
            else:
                specificity = 'None'
            if sensitivity!='None' and specificity!='None':
                score = (sensitivity + specificity)/2
                score = round(score,2)
            else:
                score = 'None'
            print('*[测试结果] 片段识别 ' + str_dataset + ' 敏感度：', sensitivity, '%, 特异度： ',specificity, '%, 得分： ',score)
            if(out_report == True):
                txt = '*[测试结果] 片段识别 ' + str_dataset + ' 敏感度：'+ sensitivity + '%, 特异度： '+ specificity + '%, 得分： '+score
                txt_obj.write(txt)
                txt_obj.close()

        except StopIteration:
            print('[Error] Model Test Error. please check data format.')

    def Predict(self, data_input):
        '''
        预测结果
        返回语音识别后的拼音符号列表
        '''
        x_in = data_input.reshape( 1, data_input.shape[0],data_input.shape[1],data_input.shape[2] )
           
        base_pred = self.model_data.predict(x=x_in)
        pred_class = np.argmax(base_pred, axis = 1)
        pred_class = pred_class.tolist()
        return pred_class

        pass
    

if(__name__=='__main__'):

    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    #进行配置，使用70%的GPU
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.93
    #config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
    set_session(tf.Session(config=config))


    datapath = ''
    modelpath = 'model_set'


    if(not os.path.exists(modelpath)): # 判断保存模型的目录是否存在
        os.makedirs(modelpath) # 如果不存在，就新建一个，避免之后保存模型的时候炸掉

    system_type = plat.system() # 由于不同的系统的文件路径表示不一样，需要进行判断
    if(system_type == 'Windows'):
        datapath = 'E:\\语音数据集'
        modelpath = modelpath + '\\'
    elif(system_type == 'Linux'):
        datapath = 'dataset'
        modelpath = modelpath + '/'
    else:
        print('*[Message] Unknown System\n')
        datapath = 'dataset'
        modelpath = modelpath + '/'

    ms = ModelSpeech(datapath)

