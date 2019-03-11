#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
to test the model performance on respective the same and different environment dataset.
"""
import os
import tensorflow as tf
import numpy as np
import random
import time
from tqdm import tqdm
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from help_func.utilities_keras import focal_loss
from general_func.gen_func import Comapare2
from readdata_bowel import Testing
from readdata_bowel import AUDIO_LENGTH, AUDIO_FEATURE_LENGTH

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#进行配置，使用90%的GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
set_session(tf.Session(config=config))


#1. define the datapath, including the same environment(test1) and the separate environment(test2)
test_same = ['/home/zhaok14/example/PycharmProjects/setsail/individual_spp/bowelsounds/perfect/test-same/','same environment']
test_distinct = ['/home/zhaok14/example/PycharmProjects/setsail/individual_spp/bowelsounds/perfect/test-distinct/','distinct environment']

training = ['/home/zhaok14/example/PycharmProjects/setsail/individual_spp/bowelsounds/perfect/train/','same environment']
validation = ['/home/zhaok14/example/PycharmProjects/setsail/individual_spp/bowelsounds/perfect/validation/','distinct environment']

test_similar = ['/home/zhaok14/example/PycharmProjects/setsail/individual_spp/bowelsounds/perfect/test-similar/','similar environment']

#2. prepare the model
modelpath = os.path.join(os.getcwd(), 'network&&weights', 'mfcc','lstm','mfcc_lstm.h5')
# model = load_model(modelpath,custom_objects={'focal_loss': focal_loss,'focal_loss_fixed': focal_loss()})
model = load_model(modelpath)

#3. follow the routine
data = Testing(training[0], validation[0])
# data = Testing(test_same[0], test_distinct[0])
choice = 'Same'
num_data = data.DataNum[choice]  # 获取数据的数量
ran_num = random.randint(0, num_data - 1)  # 获取一个随机数
overall_p = 0
overall_n = 0
overall_tp = 0
overall_tn = 0
accuracy = 0
sensitivity = 0
specificity = 0
score = 0

nowtime = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
start = time.time()
# data_count = 200
for i in tqdm(range(num_data)):
	data_input, data_labels = data.GetData((ran_num + i) % num_data, dataType=choice)  # 从随机数开始连续向后取一定数量数据
	predictions = []
	if len(data_input) <= AUDIO_LENGTH:
		data_in = np.zeros((1, AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, 1), dtype=np.float)
		data_in[0, 0:len(data_input)] = data_input
		data_pre = model.predict_on_batch(data_in)
		predictions = np.argmax(data_pre[0], axis=0)
	else:
		assert(0)

	tp, fp, tn, fn = Comapare2(predictions, data_labels[0])  # 计算metrics
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
# print('*[测试结果] 片段识别 ' + str_dataset + ' 敏感度：', sensitivity, '%, 特异度： ', specificity, '%, 得分： ', score, ', 准确度： ', accuracy, '%, 用时: ', dtime, 's.')
strg = '*[测试结果] 片段类型【{0}】 敏感度：{1}%, 特异度： {2}%, 得分： {3}, 准确度： {4}%, 用时: {5}s.'.format(choice, sensitivity, specificity, score,accuracy, dtime)
tqdm.write(strg)







