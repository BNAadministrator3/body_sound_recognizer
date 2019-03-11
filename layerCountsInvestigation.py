import platform as plat
import os
import time
import tensorflow as tf
import keras.backend as k
from tqdm import tqdm

from keras.layers import *
from keras import optimizers
from keras.regularizers import l2
from keras.models import Model
from readdata_bowel import DataSpeech

from help_func.utilities_keras import focal_loss, ReguBlock
from readdata_bowel import AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, CLASS_NUM

from Model44_trainSetMain  import ModelSpeech
class layerCurve(ModelSpeech):
    def __init__(self, datapath, layercounts):
        self.datapath = datapath
        self.slash = ''
        system_type = plat.system()  # 由于不同的系统的文件路径表示不一样，需要进行判�?
        if (system_type == 'Windows'):
            self.slash = '\\'  # 反斜�?
        elif (system_type == 'Linux'):
            self.slash = '/'  # 正斜�?
        else:
            print('*[Message] Unknown System\n')
            self.slash = '/'  # 正斜�?
        if (self.slash != self.datapath[-1]):  # 在目录路径末尾增加斜�?
            self.datapath = self.datapath + self.slash
        self.modelname = ''
        # self.model =  self.regularCNNForest(input_shape=(AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, 1),classes=CLASS_NUM, layer_counts=layercounts)
        self.model = self.bnRegularCNN(input_shape=(AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, 1), classes=CLASS_NUM)
        self.metrics = {'type': 'eval', 'sensitivity': 0, 'specificity': 0, 'score': 0, 'accuracy': 0}
        if len(self.modelname) != 0:
            self.baseSavPath = []
            if AUDIO_FEATURE_LENGTH == 26:
                strname = 'mfcc_' + self.modelname
                self.baseSavPath.append(os.path.join(os.getcwd(), 'network&&weights', 'mfcc', self.modelname, strname))
                strname = 'mfcc_' + self.modelname + '_weights'
                self.baseSavPath.append(os.path.join(os.getcwd(), 'network&&weights', 'mfcc', self.modelname, strname))
            else:
                strname = 'spec_' + self.modelname
                self.baseSavPath.append(
                    os.path.join(os.getcwd(), 'network&&weights', 'spectrogram', self.modelname, strname))
                strname = 'spec_' + self.modelname + '_weights'
                self.baseSavPath.append(
                    os.path.join(os.getcwd(), 'network&&weights', 'spectrogram', self.modelname, strname))
        else:
            assert (0)

    def bnRegularCNN(self, input_shape, classes):
        X_input = Input(name='the_input', shape=input_shape)
        level_h1 = ReguBlock(32)(X_input)
        level_m1 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h1)  # 池化层
        level_h2 = ReguBlock(64)(level_m1)
        level_m2 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h2)  # 池化层
        level_h3 = ReguBlock(128)(level_m2)
        level_m3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h3)  # 池化层

        flayer = GlobalAveragePooling2D()(level_m3)
        fc = Dense(classes, use_bias=True, kernel_initializer='he_normal')(flayer)  # 全连接层
        y_pred = Activation('softmax', name='Activation0')(fc)

        model = Model(inputs=X_input, outputs=y_pred)
        optimizer = optimizers.Adadelta()
        model.compile(optimizer=optimizer, loss=[focal_loss(alpha=0.25, gamma=2)])
        print('Regular toy cnn model within BN layers estabished.')
        self.modelname = 'layerInv'
        return model

    def regularCNNForest(self, input_shape, classes, layer_counts):
        X_input = Input(name='the_input', shape=input_shape)

        layer_h1 = Conv2D(32, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(X_input)
        layer_h2 = Conv2D(32, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(layer_h1)
        layer_p2 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h2)
        layer_h3 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(layer_p2)
        layer_h4 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(layer_h3)
        layer_p4 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h4)
        layer_h5 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(layer_p4)
        layer_h6 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(layer_h5)
        layer_p6 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h6)
        layer_h7 = Conv2D(256, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=None)(layer_p6)
        layer_h8 = Conv2D(256, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=None)(layer_h7)
        layer_p8 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h8)
        layer_h9 = Conv2D(512, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=None)(layer_p8)
        layer_h10 = Conv2D(512, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=None)(layer_h9)
        # layer_p10 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h10)
        layers = [layer_p2,layer_p4,layer_p6,layer_p8,layer_h10]
        if layer_counts in (2,4,6,8,10):
            output = layers[layer_counts//2-1]
        else:
            print('[ERROR]unresolved parameters: layer_counts.')
            assert(0)
        flayer = GlobalAveragePooling2D()(output)
        fc2 = Dense(classes, use_bias=True, kernel_initializer='he_normal')(flayer)  # 全连接层
        y_pred = Activation('softmax', name='Activation0')(fc2)

        model = Model(inputs=X_input, outputs=y_pred)
        optimizer = optimizers.Adadelta()
        model.compile(optimizer=optimizer, loss=[focal_loss(alpha=0.25, gamma=2)])
        print('Regular cnn model with {} layers estabished.'.format(layer_counts))
        self.modelname = 'layerInv'
        return model

    def TrainModel(self, datapath,epoch=10, batch_size=32):
        assert (batch_size % CLASS_NUM == 0)
        data = DataSpeech(datapath, 'train')
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
            yielddatas = data.data_genetator(batch_size,epoch)
            pbar = tqdm(yielddatas)
            for input, labels in pbar:
                stime = time.time()
                loss = self.model.train_on_batch(input[0],labels)
                # temp = self.model.predict_on_batch(input[0])
                dtime = time.time() - stime
                duration = duration + dtime
                # okay = self.model.predict_on_batch(input[0])
                # compare = self.interlayer.predict_on_batch(input[0])
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
            if i % 1 == 0:
                tmetrics = self.TestModel(sess=sess, datapath=datapath, str_dataset='train', data_count=1000, out_report=False, writer=train_writter, step=i)
                metrics = self.TestModel(sess=sess, datapath=datapath, str_dataset='eval', data_count=-1, out_report=False, writer=train_writter, step=i)
                if i > 0:
                    if metrics['score'] >= best_score:
                        self.metrics = metrics
                        self.metrics['epoch'] = i
                        best_score = metrics['score']
                        self.savpath = []
                        self.savpath.append((self.baseSavPath[0] + '_epoch' + str(i) + '.h5'))
                        self.savpath.append((self.baseSavPath[1] + '_epoch' + str(i) + '.h5'))
                        self.model.save(self.savpath[0])
                        self.model.save_weights(self.savpath[1])
        if 'epoch' in self.metrics.keys():
            print('The best metric after restriction took place in the epoch: ', self.metrics['epoch'])
            print('Sensitivity: {}; Specificity: {}; Score: {}; Accuracy: {}'.format(self.metrics['sensitivity'],self.metrics['specificity'],self.metrics['score'],self.metrics['accuracy']))
            self.TestGenerability(weightspath=self.savpath[1])
        else:
            print('The restricted best metric is not found. Done!')
            # path_test = '/home/zhaok14/example/PycharmProjects/setsail/individual_spp/network&&weights/spectrogram/mlp/spec_mlp_weights_epoch12.h5'
            # self.TestGenerability(weightspath=path_test)
        print('Training duration: {}s'.format(round(duration,2)))

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
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth=True   #不全部占满显�? 按需分配
set_session(tf.Session(config=config))
datapath = '/home/zhaok14/example/PycharmProjects/setsail/individual_spp/bowelsounds/perfect'
TEST = True
if TEST == True:
    # ns = layerCurve(datapath, 8)
    # ns.model.summary()
    ns = layerCurve(datapath, 6)
    # ns.model.summary()
    ns.TrainModel(datapath, epoch=20, batch_size=32)
else:

    sys.stdout = logger()
    for lcounts in (2,4,6,8,10):
        ns = layerCurve(datapath,lcounts)
        ns.TrainModel(datapath, epoch = 20, batch_size = 32)
        gc.collect()

