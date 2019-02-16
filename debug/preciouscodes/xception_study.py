# This routine is given credit to https://github.com/yanchummar/xception-keras/blob/master/xception_model.py
from keras.models import Model
from keras import layers
from keras.layers import Dense, Input, BatchNormalization, Activation, Flatten
from keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
import sys
sys.path.append('../../')
from readdata_bowel import AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, CLASS_NUM
from keras.utils.data_utils import get_file

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5'

def Xception():

	# Determine proper input shape
	input_shape = (AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, 1)

	img_input = Input(shape=input_shape)

	# Block 1
	x = Conv2D(32, (3, 3), padding='same', use_bias=False)(img_input)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Conv2D(32, (3, 3), padding='same', use_bias=False)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	# Block 2
	residual = Conv2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
	residual = BatchNormalization()(residual)

	x = SeparableConv2D(64, (3, 3), padding='same', use_bias=False)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = SeparableConv2D(64, (3, 3), padding='same', use_bias=False)(x)
	x = BatchNormalization()(x)

	# Block 2 Pool
	x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
	x = layers.add([x, residual])

	# Block 3
	residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
	residual = BatchNormalization()(residual)

	x = Activation('relu')(x)
	x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
	x = BatchNormalization()(x)

	# Block 3 Pool
	x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
	x_temp = layers.add([x, residual])   #size:(31,7,128)

	# Block 4
	residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
	residual = BatchNormalization()(residual)

	x = Activation('relu')(x)
	x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
	x = BatchNormalization()(x)

	x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
	x_temp = layers.add([x, residual])
	#
	# # Block 5 - 12
	# for i in range(8):
	# 	residual = x
	#
	# 	x = Activation('relu')(x)
	# 	x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
	# 	x = BatchNormalization()(x)
	# 	x = Activation('relu')(x)
	# 	x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
	# 	x = BatchNormalization()(x)
	# 	x = Activation('relu')(x)
	# 	x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
	# 	x = BatchNormalization()(x)
	#
	# 	x = layers.add([x, residual])
	#
	# residual = Conv2D(1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
	# residual = BatchNormalization()(residual)
	#
	# # Block 13
	# x = Activation('relu')(x)
	# x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
	# x = BatchNormalization()(x)
	# x = Activation('relu')(x)
	# x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False)(x)
	# x = BatchNormalization()(x)
	#
	# # Block 13 Pool
	# x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
	# x = layers.add([x, residual])
	#
	# # Block 14
	# x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False)(x)
	# x = BatchNormalization()(x)
	# x = Activation('relu')(x)
	#
	# # Block 14 part 2
	# x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False)(x)
	# x = BatchNormalization()(x)
	# x = Activation('relu')(x)
	#
	# # Fully Connected Layer
	# x = GlobalAveragePooling2D()(x)



	x_temp = Flatten()(x_temp)
	x = Dense(CLASS_NUM, activation='softmax')(x_temp)
	inputs = img_input

	# Create model
	model = Model(inputs, x, name='xception')

	# Download and cache the Xception weights file
	# weights_path = get_file('xception_weights.h5', WEIGHTS_PATH, cache_subdir='models')

	# load weights
	# model.load_weights(weights_path)

	return model


"""
	Instantiate the model by using the following line of code
	
"""
model = Xception()
print('inception:', model.summary())