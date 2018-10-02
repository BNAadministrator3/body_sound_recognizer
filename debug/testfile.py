from general_func.file_wav import GetFrequencyFeature4
from general_func.file_wav import GetFrequencyFeature3
from general_func.file_wav import GetFrequencyFeatures
from general_func.file_wav import read_wav_data
import matplotlib.pyplot as plt
import tensorflow as tf

str = 'E:\workspace\pycharmworkspace\ASRT_SpeechRecognition-0.3\dataset\data_thchs30\data\A2_0.wav'
str2 = 'E:\workspace\Data\\rs_sdegrom\ICBHI_final_database_downsample\\train\\103_2b2_Ar_mc_LittC2SE.wav'
str3 = 'E:\workspace\Data\\rs_sdegrom\ICBHI_final_database_downsample\\train\\107_2b3_Pr_mc_AKGC417L.wav'

signal, fs = read_wav_data(str2)

spectrogram = GetFrequencyFeatures(signal,fs)
# spectrogram = GetFrequencyFeature4(signal,fs,True)

plt.pcolormesh(spectrogram,cmap='gray_r')
plt.colorbar()
plt.ylabel('Frequency[Hz]')
plt.xlabel('Time[sec]')
plt.show()


def CreateModel(self):
    self.input_data = tf.placeholder(dtype=tf.float32, shape=[None, self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH, 1])
    self.label = tf.placeholder(dtype=tf.float32, shape=[None, 4])
    self.is_train = tf.placeholder(dtype=tf.bool)

    conv2d_1 = tf.layers.conv2d(self.input_data, 32, (3, 3), use_bias=True, padding='same',
                                kernel_initializer=tf.keras.initializers.he_normal(), name='conv2d_1')
    bn_1 = self.batch_norm(conv2d_1, self.is_train, scope='bn_1')
    relu_1 = tf.keras.activations.relu(bn_1)
    droput_1 = tf.layers.dropout(relu_1, rate=0.1, training=self.is_train)
    conv2d_2 = tf.layers.conv2d(droput_1, 32, (3, 3), use_bias=True, padding='same',
                                kernel_initializer=tf.keras.initializers.he_normal(), name='conv2d_2')
    bn_2 = self.batch_norm(conv2d_2, self.is_train, scope='bn_2')
    relu_2 = tf.keras.activations.relu(bn_2)
    max_pool_2 = tf.layers.max_pooling2d(relu_2, pool_size=2, strides=2, padding="valid", name='max_pool_2')
    droput_2 = tf.layers.dropout(max_pool_2, rate=0.1, training=self.is_train)

    conv2d_3 = tf.layers.conv2d(droput_2, 64, (3, 3), use_bias=True, padding='same',
                                kernel_initializer=tf.keras.initializers.he_normal(), name='conv2d_3')  # 卷积层
    bn_3 = self.batch_norm(conv2d_3, self.is_train, scope='bn_3')
    relu_3 = tf.keras.activations.relu(bn_3)
    droput_3 = tf.layers.dropout(relu_3, rate=0.2, training=self.is_train)  # 随机丢失层
    conv2d_4 = tf.layers.conv2d(droput_3, 64, (3, 3), use_bias=True, padding='same',
                                kernel_initializer=tf.keras.initializers.he_normal(), name='conv2d_4')  # 卷积层
    bn_4 = self.batch_norm(conv2d_4, self.is_train, scope='bn_4')
    relu_4 = tf.keras.activations.relu(bn_4)
    max_pool_4 = tf.layers.max_pooling2d(relu_4, pool_size=2, strides=2, padding="valid", name='max_pool_4')
    droput_4 = tf.layers.dropout(max_pool_4, rate=0.2, training=self.is_train)

    conv2d_5 = tf.layers.conv2d(droput_4, 32, (3, 3), use_bias=True, padding='same',
                                kernel_initializer=tf.keras.initializers.he_normal(), name='conv2d_5')
    bn_5 = self.batch_norm(conv2d_5, self.is_train, scope='bn_5')
    relu_5 = tf.keras.activations.relu(bn_5)
    droput_5 = tf.layers.dropout(relu_5, rate=0.3, training=self.is_train)
    conv2d_6 = tf.layers.conv2d(droput_5, 32, (3, 3), use_bias=True, padding='same',
                                kernel_initializer=tf.keras.initializers.he_normal(), name='conv2d_6')
    bn_6 = self.batch_norm(conv2d_6, self.is_train, scope='bn_6')
    relu_6 = tf.keras.activations.relu(bn_6)
    max_pool_6 = tf.layers.max_pooling2d(relu_6, pool_size=2, strides=2, padding="valid", name='max_pool_6')
    droput_6 = tf.layers.dropout(max_pool_6, rate=0.3, training=self.is_train)

    conv2d_7 = tf.layers.conv2d(droput_6, 32, (3, 3), use_bias=True, padding='same',
                                kernel_initializer=tf.keras.initializers.he_normal(), name='conv2d_7')
    bn_7 = self.batch_norm(conv2d_7, self.is_train, scope='bn_7')
    relu_7 = tf.keras.activations.relu(bn_7)
    droput_7 = tf.layers.dropout(relu_7, rate=0.4, training=self.is_train)
    conv2d_8 = tf.layers.conv2d(droput_7, 32, (3, 3), use_bias=True, padding='same',
                                kernel_initializer=tf.keras.initializers.he_normal(), name='conv2d_8')
    bn_8 = self.batch_norm(conv2d_8, self.is_train, scope='bn_8')
    relu_8 = tf.keras.activations.relu(bn_8)

    flat_layer = tf.layers.flatten(relu_8)

    droput_9 = tf.layers.dropout(flat_layer, rate=0.4, training=self.is_train)
    fc_1 = tf.layers.dense(droput_9, 128, activation=tf.keras.activations.relu, use_bias=True,
                           kernel_initializer=tf.keras.initializers.he_normal(),
                           name='fc_1')
    droput_10 = tf.layers.dropout(flat_layer, rate=0.5, training=self.is_train)
    fc_2 = tf.layers.dense(droput_10, self.MS_OUTPUT_SIZE, activation=tf.keras.activations.relu, use_bias=True,
                           kernel_initializer=tf.keras.initializers.he_normal(),
                           name='fc_2')

    y_pred = tf.keras.activations.softmax(fc_2)
    self.y_pred = tf.clip_by_value(y_pred, 1e-10, 1.0)
    # self.loss = -tf.reduce_mean(self.label * tf.log(self.y_pred) )
    self.loss = - tf.reduce_sum(self.label * tf.log(self.y_pred))
    tf.summary.scalar('loss', self.loss)

    self.optimize = tf.train.AdadeltaOptimizer(learning_rate=0.01, rho=0.95, epsilon=1e-06).minimize(self.loss)

    print(input[1])
    c1 = 0
    c2 = 0
    c3 = 0
    c4 = 0
    for x in range(16):
        if input[1][x][0]==1:
            c1=c1+1
        if input[1][x][1]==1:
            c2=c2+1
        if input[1][x][2]==1:
            c3=c3+1
        if input[1][x][3]==1:
            c4=c4+1
    print('c1: ', c1)
    print('c2: ', c2)
    print('c3: ', c3)
    print('c4: ', c4)
