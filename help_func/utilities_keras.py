from keras import backend as K
from keras.layers import *
from keras.activations import relu
from keras.regularizers import l2

def focal_loss(gamma=2, alpha=0.25):

    def focal_loss_fixed(y_true, y_pred):#with tensorflow
        # Define epsilon so that the backpropagation will not result in NaN
        # for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediction value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate cross entropy
        cross_entropy = -y_true * K.log(y_pred)
        # Calculate weight that consists of  modulating factor and weighting factor
        weight = alpha * y_true * K.pow((1 - y_pred), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = 10.0 * K.sum(loss, axis=1)
        return loss

    return focal_loss_fixed


# The codes reference to the https://github.com/km1414/CNN-models/blob/master/googlenet-lite/googlenet-lite.py
# Inception module - major building block of the googlenet
def inception_model(input, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj):
    #1*1 convolutions
    conv_1x1 = Conv2D(filters=filters_1x1, kernel_size=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(input)
    #3*3 convolutions
    conv_3x3_reduce = Conv2D(filters=filters_3x3_reduce, kernel_size=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(input)
    conv_3x3 = Conv2D(filters=filters_3x3, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01))(conv_3x3_reduce)
    #5*5 convolutions
    conv_5x5_reduce  = Conv2D(filters=filters_5x5_reduce, kernel_size=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(input)
    conv_5x5 = Conv2D(filters=filters_5x5, kernel_size=(5, 5), padding='same', activation='relu', kernel_regularizer=l2(0.01))(conv_5x5_reduce)
    #1*1 convolutions with maxpooling
    maxpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input)
    maxpool_proj = Conv2D(filters=filters_pool_proj, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(maxpool)

    inception_output = concatenate([conv_1x1, conv_3x3, conv_5x5, maxpool_proj], axis=3)  # use tf as backend

    return inception_output

def inception_separable(input, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj):
    #1*1 convolutions
    conv_1x1 = Conv2D(filters=filters_1x1, kernel_size=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(input)
    #3*3 convolutions
    conv_3x3_reduce = Conv2D(filters=filters_3x3_reduce, kernel_size=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(input)
    conv_3x3 = Conv2D(filters=filters_3x3, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01))(conv_3x3_reduce)
    #5*5 convolutions
    conv_5x5_reduce  = Conv2D(filters=filters_5x5_reduce, kernel_size=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(input)
    conv_5x5 = SeparableConv2D(filters=filters_5x5, kernel_size=(5, 5), padding='same', activation='relu',kernel_regularizer=l2(0.01))(conv_5x5_reduce)
    #1*1 convolutions with maxpooling
    maxpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input)
    maxpool_proj = Conv2D(filters=filters_pool_proj, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(maxpool)

    inception_output = concatenate([conv_1x1, conv_3x3, conv_5x5, maxpool_proj], axis=3)  # use tf as backend

    return inception_output

def inception_Xseparable(input, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj):
    #1*1 convolutions
    conv_1x1 = Conv2D(filters=filters_1x1, kernel_size=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(input)
    #3*3 convolutions
    conv_3x3_reduce = Conv2D(filters=filters_3x3_reduce, kernel_size=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(input)
    conv_3x3 = SeparableConv2D(filters=filters_3x3, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01))(conv_3x3_reduce)
    #5*5 convolutions
    conv_5x5_reduce  = Conv2D(filters=filters_5x5_reduce, kernel_size=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(input)
    conv_5x5 = SeparableConv2D(filters=filters_5x5, kernel_size=(5, 5), padding='same', activation='relu',kernel_regularizer=l2(0.01))(conv_5x5_reduce)
    #1*1 convolutions with maxpooling
    maxpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input)
    maxpool_proj = Conv2D(filters=filters_pool_proj, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(maxpool)

    inception_output = concatenate([conv_1x1, conv_3x3, conv_5x5, maxpool_proj], axis=3)  # use tf as backend

    return inception_output



#The codes reference to the https://www.kaggle.com/meownoid/tiny-resnet-with-keras-99-314
def block(n_output, upscale=True):
    # n_output: number of feature maps in the block
    # upscale: should we use the 1x1 conv2d mapping for shortcut or not

    # keras functional api: return the function of type
    # Tensor -> Tensor
    def f(x):

        # H_l(x):
        # first pre-activation
        h = BatchNormalization()(x)
        h = Activation(relu)(h)
        # first convolution
        h = Conv2D(kernel_size=3, filters=n_output, strides=1, padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(h)

        # second pre-activation
        h = BatchNormalization()(h)
        h = Activation(relu)(h)
        # second convolution
        h = Conv2D(kernel_size=3, filters=n_output, strides=1, padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(h)

        # f(x):
        if upscale:
            # 1x1 conv2d
            f = Conv2D(kernel_size=1, filters=n_output, strides=1, padding='same')(x)
        else:
            # identity
            f = x

        # F_l(x) = f(x) + H_l(x):
        return add([f, h])

    return f