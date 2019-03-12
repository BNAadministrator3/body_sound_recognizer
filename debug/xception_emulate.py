# This routine is given credit to https://github.com/yanchummar/xception-keras/blob/master/xception_model.py
from keras.models import Model
from keras import layers
from keras.layers import Dense, Input, BatchNormalization, Activation, Flatten
from keras.layers import Conv2D, SeparableConv2D
import sys

sys.path.append('../../')
from release.readdata_bowel import AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, CLASS_NUM

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5'


def Xception():
    # Determine proper input shape
    input_shape = (AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, 1)

    img_input = Input(shape=input_shape)

    # Block 1
    x = Conv2D(32, (3, 3), use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(residual)

    for i in range(1):
    	residual = x

    	x = Activation('relu')(x)
    	x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    	x = BatchNormalization()(x)
    	x = Activation('relu')(x)
    	x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    	x = BatchNormalization()(x)
    	x = Activation('relu')(x)
    	x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    	x = BatchNormalization()(x)

    	x = layers.add([x, residual])

    x_temp = x

    x_temp = Flatten()(x_temp)
    x = Dense(CLASS_NUM, activation='softmax')(x_temp)
    inputs = img_input

    # Create model
    model = Model(inputs, x, name='xception')


    return model


"""
	Instantiate the model by using the following line of code

"""
model = Xception()
print('inception:', model.summary())